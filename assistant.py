"""Conversational OpenAI settings assistant for Yurameki."""

import ctypes
from ctypes import wintypes
import json
import queue
import threading
import urllib.error
import urllib.request

import bpy
from bpy.props import StringProperty
from bpy.types import Operator


_CREDENTIAL_TARGET = "Yurameki/OpenAI API Key"
_MODEL = "gpt-5.4-nano"
_API_URL = "https://api.openai.com/v1/responses"
_MAX_HISTORY = 20

_PARAMETERS = {
    "yurameki_root_locked_points": ("Root Locked Points", "int", 1, 128, "root joints fixed to the animated pose"),
    "yurameki_gravity": ("Gravity m/s2", "vector", -1000.0, 1000.0, "XYZ acceleration; normally [0, 0, -9.81]"),
    "yurameki_damping": ("Damping", "float", 0.0, 0.99, "velocity damping; higher settles motion sooner"),
    "yurameki_max_velocity_mps": ("Max Velocity m/s", "float", 0.0, 100.0, "speed limit; zero disables it"),
    "yurameki_particle_mass_g": ("Particle Mass g", "float", 0.001, 1000.0, "mass per free joint"),
    "yurameki_iterations": ("Iterations", "int", 1, 256, "distance and bend constraint iterations"),
    "yurameki_stretch_compliance_log10": ("Stretch Compliance log10", "float", -12.0, 0.0, "higher is stretchier; -2 means 1e-2"),
    "yurameki_bend_compliance_log10": ("Bend Compliance log10", "float", -12.0, 0.0, "higher bends more easily; -5 means 1e-5"),
    "yurameki_collision_margin_mm": ("Collision Margin mm", "float", 0.01, 100.0, "target collider separation"),
    "yurameki_collision_search_mm": ("Collision Search mm", "float", 0.1, 1000.0, "nearest-surface search radius"),
    "yurameki_collision_max_correction_mm": ("Collision Max Correction mm", "float", 0.01, 500.0, "maximum push-out per pass"),
    "yurameki_collision_response": ("Collision Response", "float", 0.0, 1.0, "fraction of position correction applied"),
    "yurameki_collision_velocity_damping": ("Collision Velocity Damping", "float", 0.0, 1.0, "velocity removed at contact"),
    "yurameki_collision_passes": ("Collision Passes", "int", 1, 32, "segment collision passes"),
    "yurameki_post_collision_iterations": ("Post Collision Iterations", "int", 0, 64, "constraint/collision reconciliation passes"),
    "yurameki_auto_substep_mm": ("Auto Substep mm", "float", 0.05, 100.0, "maximum motion per automatic substep; smaller is safer and slower"),
    "yurameki_max_substeps": ("Max Substeps", "int", 1, 512, "automatic substep cap"),
}

_messages: list[dict[str, str]] = []
_snapshots: list[dict[str, object]] = []
_result_queue: queue.Queue = queue.Queue()
_busy = False
_timer_registered = False


class _CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
        ("Attributes", wintypes.LPVOID), ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def _advapi32():
    dll = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    dll.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(_CREDENTIALW))]
    dll.CredReadW.restype = wintypes.BOOL
    dll.CredWriteW.argtypes = [ctypes.POINTER(_CREDENTIALW), wintypes.DWORD]
    dll.CredWriteW.restype = wintypes.BOOL
    dll.CredFree.argtypes = [wintypes.LPVOID]
    return dll


def get_api_key() -> str | None:
    credential = ctypes.POINTER(_CREDENTIALW)()
    dll = _advapi32()
    if not dll.CredReadW(_CREDENTIAL_TARGET, 1, 0, ctypes.byref(credential)):
        return None
    try:
        size = int(credential.contents.CredentialBlobSize)
        if not size:
            return None
        raw = ctypes.string_at(credential.contents.CredentialBlob, size)
        return raw.decode("utf-16-le")
    finally:
        dll.CredFree(credential)


def set_api_key(api_key: str) -> None:
    value = api_key.strip()
    if not value:
        raise ValueError("API key is empty")
    raw = value.encode("utf-16-le")
    blob = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
    credential = _CREDENTIALW()
    credential.Type = 1  # CRED_TYPE_GENERIC
    credential.TargetName = _CREDENTIAL_TARGET
    credential.CredentialBlobSize = len(raw)
    credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
    credential.Persist = 2  # CRED_PERSIST_LOCAL_MACHINE
    credential.UserName = "OpenAI"
    if not _advapi32().CredWriteW(ctypes.byref(credential), 0):
        raise ctypes.WinError(ctypes.get_last_error())


def has_api_key() -> bool:
    try:
        return bool(get_api_key())
    except Exception:
        return False


def messages():
    return tuple(_messages)


def is_busy() -> bool:
    return _busy


def _snapshot(wm) -> dict[str, object]:
    result = {}
    for name, (_label, kind, _minimum, _maximum, _description) in _PARAMETERS.items():
        value = getattr(wm, name)
        result[name] = list(value) if kind == "vector" else value
    return result


def _apply_snapshot(wm, values: dict[str, object]) -> None:
    for name, value in values.items():
        if name in _PARAMETERS:
            setattr(wm, name, value)


def _append(role: str, text: str) -> None:
    _messages.append({"role": role, "text": text.strip()})
    del _messages[:-_MAX_HISTORY]


def undo(wm) -> bool:
    if not _snapshots:
        _append("assistant", "戻せる設定履歴がまだありません。")
        return False
    _apply_snapshot(wm, _snapshots.pop())
    _append("assistant", "一つ前の設定に戻しました。")
    wm.yurameki_assistant_status = "Previous settings restored"
    return True


def clear_conversation(wm) -> None:
    _messages.clear()
    _snapshots.clear()
    wm.yurameki_assistant_status = "Conversation cleared"


def _parameter_reference(current: dict[str, object]) -> list[dict]:
    return [
        {"name": name, "label": spec[0], "type": spec[1], "min": spec[2], "max": spec[3], "description": spec[4], "current": current[name]}
        for name, spec in _PARAMETERS.items()
    ]


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "A short Japanese response, about one sentence."},
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": list(_PARAMETERS)},
                        "value": {"anyOf": [{"type": "number"}, {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}]},
                    },
                    "required": ["name", "value"], "additionalProperties": False,
                },
            },
        },
        "required": ["message", "changes"], "additionalProperties": False,
    }


def _request_worker(api_key: str, instruction: str, current: dict, history: list[dict]) -> None:
    system = (
        "You are Yurameki Assistant, a careful NVIDIA Warp long-hair simulation tuning expert. "
        "Answer in Japanese. Change only parameters needed for the user's request. "
        "Never exceed documented ranges. If the request is ambiguous, ask one short question and return no changes. "
        "Compliance values are log10: larger values are softer. Preserve collision safety unless explicitly requested."
    )
    context = {
        "parameter_reference": _parameter_reference(current),
        "recent_conversation": history[-8:],
        "user_instruction": instruction,
    }
    payload = {
        "model": _MODEL,
        "instructions": system,
        "input": json.dumps(context, ensure_ascii=False),
        "reasoning": {"effort": "low"},
        "max_output_tokens": 800,
        "text": {"format": {"type": "json_schema", "name": "yurameki_settings", "strict": True, "schema": _schema()}},
    }
    request = urllib.request.Request(
        _API_URL, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = body.get("output_text")
        if not text:
            for item in body.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text = content.get("text")
                        break
        if not text:
            raise ValueError("The API returned no text")
        _result_queue.put((True, json.loads(text)))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("error", {}).get("message", detail)
        except Exception:
            pass
        _result_queue.put((False, f"OpenAI API error ({exc.code}): {detail}"))
    except Exception as exc:
        _result_queue.put((False, f"Assistant request failed: {exc}"))


def _validated_changes(data) -> tuple[dict[str, object], str]:
    values = {}
    for change in data.get("changes", []):
        name = change.get("name")
        if name not in _PARAMETERS:
            continue
        _label, kind, minimum, maximum, _description = _PARAMETERS[name]
        value = change.get("value")
        if kind == "vector":
            if not isinstance(value, list) or len(value) != 3:
                continue
            values[name] = tuple(max(minimum, min(maximum, float(v))) for v in value)
        else:
            number = max(minimum, min(maximum, float(value)))
            values[name] = int(round(number)) if kind == "int" else number
    message = str(data.get("message", "設定を確認しました。")).strip()
    return values, message[:500]


def _tag_redraw() -> None:
    """Refresh visible UI regions after a background request finishes."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def _poll_result():
    global _busy, _timer_registered
    try:
        ok, result = _result_queue.get_nowait()
    except queue.Empty:
        return 0.2
    _busy = False
    _timer_registered = False
    wm = bpy.context.window_manager
    if ok:
        try:
            changes, message = _validated_changes(result)
            if changes:
                _snapshots.append(_snapshot(wm))
                del _snapshots[:-_MAX_HISTORY]
                _apply_snapshot(wm, changes)
                wm.yurameki_assistant_status = f"Applied {len(changes)} change(s)"
            else:
                wm.yurameki_assistant_status = "No settings changed"
            _append("assistant", message)
        except Exception as exc:
            wm.yurameki_assistant_status = f"Invalid assistant response: {exc}"
            _append("assistant", "返答を設定へ反映できませんでした。")
    else:
        wm.yurameki_assistant_status = str(result)[:500]
        _append("assistant", "通信に失敗しました。ステータスを確認してください。")
    _tag_redraw()
    return None


def start_request(wm, instruction: str) -> None:
    global _busy, _timer_registered
    text = instruction.strip()
    if not text:
        raise ValueError("Enter an instruction first")
    if _busy:
        raise RuntimeError("Assistant is already thinking")
    if not bpy.app.online_access:
        raise RuntimeError("Online Access is disabled in Blender Preferences")
    key = get_api_key()
    if not key:
        raise RuntimeError("Register an OpenAI API key first")
    _append("user", text)
    current = _snapshot(wm)
    history = list(_messages[:-1])
    _busy = True
    wm.yurameki_assistant_status = "Thinking..."
    thread = threading.Thread(target=_request_worker, args=(key, text, current, history), daemon=True)
    thread.start()
    if not _timer_registered:
        bpy.app.timers.register(_poll_result, first_interval=0.2)
        _timer_registered = True


class YURAMEKI_OT_set_api_key(Operator):
    bl_idname = "yurameki.set_api_key"
    bl_label = "OpenAI API Key"
    bl_description = "Store or update the OpenAI API key in Windows Credential Manager"

    api_key: StringProperty(name="API Key", subtype="PASSWORD", options={"SKIP_SAVE"})

    def invoke(self, context, event):
        self.api_key = ""
        return context.window_manager.invoke_props_dialog(self, width=460)

    def execute(self, context):
        try:
            set_api_key(self.api_key)
            context.window_manager.yurameki_assistant_status = "API key stored in Windows Credential Manager"
            self.report({"INFO"}, "OpenAI API key stored securely")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, f"Could not store API key: {exc}")
            return {"CANCELLED"}


class YURAMEKI_OT_assistant_send(Operator):
    bl_idname = "yurameki.assistant_send"
    bl_label = "Send"
    bl_description = "Ask Yurameki Assistant to tune the current settings"

    def execute(self, context):
        wm = context.window_manager
        instruction = wm.yurameki_assistant_input
        normalized = instruction.replace(" ", "")
        if any(term in normalized for term in ("前の方がよかった", "前のほうがよかった", "一つ前に戻", "ひとつ前に戻", "元に戻")):
            _append("user", instruction)
            undo(wm)
            wm.yurameki_assistant_input = ""
            return {"FINISHED"}
        try:
            start_request(wm, instruction)
            wm.yurameki_assistant_input = ""
            return {"FINISHED"}
        except Exception as exc:
            wm.yurameki_assistant_status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class YURAMEKI_OT_assistant_undo(Operator):
    bl_idname = "yurameki.assistant_undo"
    bl_label = "Previous Settings"
    bl_description = "Restore the settings from before the last assistant change"

    def execute(self, context):
        return {"FINISHED"} if undo(context.window_manager) else {"CANCELLED"}


class YURAMEKI_OT_assistant_clear(Operator):
    bl_idname = "yurameki.assistant_clear"
    bl_label = "New Conversation"
    bl_description = "Clear the assistant conversation and its undo history"

    def execute(self, context):
        clear_conversation(context.window_manager)
        return {"FINISHED"}


CLASSES = (
    YURAMEKI_OT_set_api_key,
    YURAMEKI_OT_assistant_send,
    YURAMEKI_OT_assistant_undo,
    YURAMEKI_OT_assistant_clear,
)


def unregister_runtime() -> None:
    global _busy, _timer_registered
    _busy = False
    if _timer_registered and bpy.app.timers.is_registered(_poll_result):
        bpy.app.timers.unregister(_poll_result)
    _timer_registered = False
    _messages.clear()
    _snapshots.clear()
