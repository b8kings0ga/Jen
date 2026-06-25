from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from voice_assistant.front_note import front_note_html_to_text, sanitize_front_note_html
from voice_assistant.fallbacks import direct_fallback_response_from_tools, direct_response_from_domain_probe, no_followup_fallback_prompt, recent_tool_context_from_rows
from voice_assistant.ui import live_log_source_group
from voice_assistant.event_metadata import (
    agno_tool_output_metadata,
    plan_prefetch_completed_metadata,
    plan_prefetch_hit_metadata,
    plan_prefetch_miss_metadata,
    plan_prefetch_started_metadata,
    tool_retry_event_metadata,
    tool_started_event_metadata,
    tool_voice_summary_event_metadata,
)
from voice_assistant.local_actions import workspace_layout_rects
from voice_assistant.plan_recovery import plan_recovery_tool_args
from voice_assistant.planning import extract_weather_location, video_search_query_from_text
from voice_assistant.planning import execution_plan_from_domain_probe
from voice_assistant.pro_lane import ProLaneWorker, coding_or_debug_task_requested, tool_names_for_domain_probe
from voice_assistant.pro_guards import (
    classify_followup_action,
    should_dedupe_completed_action,
    should_stop_for_initial_answer,
    short_tool_name,
    tool_missing_requirements,
)
from voice_assistant.speech_text import split_speech_text, strip_think_blocks
from voice_assistant.speech import SpeechQueue
import voice_assistant.speech as speech_module
from voice_assistant.store import VoiceSessionStore
from voice_assistant.tool_policy import evaluate_runtime_tool_preflight, prepare_runtime_tool_call
from voice_assistant.tool_runtime import (
    callable_tool_map,
    tool_retry_backoff_seconds,
    tool_timeout_error_message,
)
from voice_assistant.tool_state import ToolTurnState
from voice_assistant.tool_registry import build_voice_tools
import voice_assistant.tool_registry as tool_registry
from voice_assistant.platformer_rule_engine import (
    DEFAULT_PHYSICS_PROFILE,
    emit_runtime_level,
    generate_collision_layer,
    generate_logic_map,
    plan_tiles,
    validate_manifest,
    validate_playability,
    validate_sockets,
)
from kenney_tile_condition_classifier import classify_asset_dir
from voice_assistant.url_utils import extract_urls_from_value, looks_like_video_url
from voice_assistant.bot import VoiceBot
import voice_assistant.asr as asr_module
from voice_assistant.config import parse_args
from voice_assistant.coding_monitor import CodingAppRunner, CodingTaskMonitor, resolve_host_venv, summarize_codex_event, summarize_coding_event, validate_workspace_static_assets
import voice_assistant.coding_monitor as coding_monitor
from voice_assistant.coding_workspace import CodingWorkspaceIndex, read_manifest
from voice_assistant.voice_text import normalize_asr_transcript
import voice_assistant.domain_probe as domain_probe
from voice_assistant.domain_probe import format_domain_probe_prompt, probe_domains
from voice_assistant.input_hotkeys import HoldKeyTap
from voice_assistant.daily_slot_parser import parse_daily_actions, split_daily_segments


class SpeechTextTests(unittest.TestCase):
    def test_strip_think_blocks_removes_closed_and_unclosed_variants(self) -> None:
        self.assertEqual(strip_think_blocks("你好<think>hidden</think>世界"), "你好世界")
        self.assertEqual(strip_think_blocks("你好<thinking>hidden</thinking>世界"), "你好世界")
        self.assertEqual(strip_think_blocks("结果<think>unfinished"), "结果")
        self.assertEqual(strip_think_blocks("结果<thinking>unfinished"), "结果")

    def test_split_speech_text_keeps_chunks_under_limit(self) -> None:
        chunks = split_speech_text("先打开浏览器，然后搜索视频，接着播放第一个结果，最后告诉我打开了。", max_chars=12)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 12 for chunk in chunks), chunks)

    def test_normalize_asr_transcript_strips_fillers_and_keeps_corrections(self) -> None:
        text = normalize_asr_transcript("嗯，那个 Chrome打开YMCA不是打开官方的MTV随便一个就行")
        self.assertEqual(text, "Chrome打开YMCA，不是打开官方的MTV，随便一个就行")

    def test_normalize_asr_transcript_does_not_drop_negative_fragment(self) -> None:
        self.assertEqual(normalize_asr_transcript("不是这个，换一个"), "不是这个，换一个")

    def test_normalize_asr_transcript_later_self_correction_overrides_target(self) -> None:
        text = normalize_asr_transcript("放一下YMCA呃,不是放一個青葉世子的MV")
        self.assertEqual(text, "放一个青叶世子的MV")
        self.assertNotIn("YMCA", text)

    def test_normalize_asr_transcript_converts_common_traditional_variants(self) -> None:
        self.assertEqual(normalize_asr_transcript("明天下雨嗎"), "明天下雨吗")
        self.assertEqual(normalize_asr_transcript("聖保羅下雨麼"), "圣保罗下雨么")
        self.assertEqual(
            normalize_asr_transcript("打開瀏覽器搜尋天氣，然後把結果寫到便籤"),
            "打开浏览器搜索天气，然后把结果写到便签",
        )
        self.assertEqual(
            normalize_asr_transcript("這個遊戲方向鍵沒有綁定，畫面不絲滑"),
            "游戏方向键没有绑定，画面不丝滑",
        )


class AsrConfigTests(unittest.TestCase):
    def test_default_asr_uses_simplified_chinese_with_mixed_names(self) -> None:
        args = parse_args([])
        self.assertEqual(args.language, "zh")
        self.assertIn("Simplified Chinese", args.asr_prompt)
        self.assertIn("Preserve English app names", args.asr_prompt)
        self.assertIn("Never output Traditional Chinese", args.asr_prompt)

    def test_auto_language_is_omitted_from_asr_request_fields(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as fp:
            fp.write(b"RIFFfake")
            fp.flush()
            captured: dict[str, object] = {}
            original_form = asr_module.multipart_form_data
            original_urlopen = asr_module.urlopen_text

            def fake_form(*, fields, files):
                captured["fields"] = dict(fields)
                return b"body", "multipart/form-data; boundary=x"

            def fake_urlopen(req, **kwargs):
                return '{"text":"open Chrome"}'

            try:
                asr_module.multipart_form_data = fake_form
                asr_module.urlopen_text = fake_urlopen
                args = SimpleNamespace(
                    asr_model="whisper-large-v3-turbo",
                    language="auto",
                    asr_prompt="keep English",
                    gjallarhorn_base_url="http://localhost:4000/v1",
                    api_key="fake",
                    asr_timeout=1.0,
                    verify_tls=False,
                )
                text = asr_module.transcribe_with_gjallarhorn(Path(fp.name), args)
            finally:
                asr_module.multipart_form_data = original_form
                asr_module.urlopen_text = original_urlopen

        self.assertEqual(text, "open Chrome")
        self.assertNotIn("language", captured["fields"])


class PlatformerRuleEngineTests(unittest.TestCase):
    def manifest(self) -> dict[str, object]:
        return tool_registry._kenney_platformer_manifest()

    def test_current_kenney_manifest_v2_validates(self) -> None:
        manifest = self.manifest()
        result = validate_manifest(manifest)
        self.assertTrue(result["ok"], result)
        self.assertEqual(manifest["version"], 2)
        self.assertIn("tiles", manifest)
        self.assertIn("autotile_rules", manifest)

    def test_kenney_manifest_tiles_include_auto_conditions(self) -> None:
        manifest = self.manifest()
        self.assertTrue(manifest["tiles"])
        for tile in manifest["tiles"]:
            self.assertIn("condition", tile)
            self.assertIn("rotations", tile)
            self.assertIn("rot0", tile["rotations"])
            self.assertIn("rot90", tile["rotations"])
            self.assertIn("rot180", tile["rotations"])
            self.assertIn("rot270", tile["rotations"])
            self.assertEqual(tile["condition_source"], "auto_edge_classifier")

    def test_kenney_condition_classifier_covers_all_base_tiles(self) -> None:
        asset_dir = Path("/Users/a1234/Downloads/kenney_pixel-platformer")
        if not asset_dir.exists():
            self.skipTest("Kenney pixel-platformer asset pack is not installed")
        classified = classify_asset_dir(asset_dir)
        self.assertEqual(len(classified["sheets"]["terrain"]), 180)
        self.assertEqual(len(classified["sheets"]["backgrounds"]), 24)
        self.assertEqual(len(classified["sheets"]["characters"]), 27)
        sample = classified["sheets"]["terrain"]["2"]
        self.assertEqual(set(sample["rotations"]), {"rot0", "rot90", "rot180", "rot270"})
        self.assertEqual(set(sample["base_condition"]), {"top", "right", "bottom", "left"})

    def test_invalid_manifest_reports_missing_socket(self) -> None:
        manifest = json.loads(json.dumps(self.manifest()))
        del manifest["tiles"][0]["sockets"]["right"]
        result = validate_manifest(manifest)
        self.assertFalse(result["ok"])
        self.assertTrue(any("sockets.right" in error for error in result["errors"]), result)

    def test_animation_frames_are_validated(self) -> None:
        manifest = json.loads(json.dumps(self.manifest()))
        manifest["animations"]["flying_enemy_flap"]["frames"] = [999]
        result = validate_manifest(manifest)
        self.assertFalse(result["ok"])
        self.assertTrue(any("animations.flying_enemy_flap.frames" in error for error in result["errors"]), result)

    def test_generate_logic_map_is_seed_deterministic_and_playable(self) -> None:
        options = {"width": 32, "height": 12, "theme": "grass", "difficulty": "easy", "seed": 7}
        first = generate_logic_map(options)
        second = generate_logic_map(options)
        self.assertEqual(first, second)
        self.assertEqual(first["grid"][first["spawn"]["y"]][first["spawn"]["x"]], "spawn")
        self.assertEqual(first["grid"][first["goal"]["y"]][first["goal"]["x"]], "goal")
        self.assertTrue(validate_playability(first, DEFAULT_PHYSICS_PROFILE)["ok"])

    def test_validate_playability_rejects_unsupported_spawn(self) -> None:
        logic = generate_logic_map({"width": 24, "height": 10, "difficulty": "easy", "seed": 1})
        sx, sy = logic["spawn"]["x"], logic["spawn"]["y"]
        logic["grid"][sy + 1][sx] = "air"
        result = validate_playability(logic, DEFAULT_PHYSICS_PROFILE)
        self.assertFalse(result["ok"])
        self.assertIn("spawn must stand on solid/platform", result["errors"])

    def test_validate_playability_rejects_unreachable_goal(self) -> None:
        logic = generate_logic_map({"width": 24, "height": 10, "difficulty": "easy", "seed": 1})
        gx, gy = logic["goal"]["x"], logic["goal"]["y"]
        logic["grid"][gy][gx] = "air"
        logic["goal"] = {"x": gx, "y": 1}
        logic["grid"][1][gx] = "goal"
        result = validate_playability(logic, DEFAULT_PHYSICS_PROFILE)
        self.assertFalse(result["ok"])

    def test_plan_tiles_uses_autotile_roles_and_sockets_validate(self) -> None:
        manifest = self.manifest()
        logic = generate_logic_map({"width": 20, "height": 10, "difficulty": "easy", "seed": 2})
        visual = plan_tiles(logic, manifest, "grass")
        floor_y = logic["height"] - 3
        self.assertEqual(visual["tiles"][floor_y][0], "ground.grass.top.left")
        self.assertEqual(visual["tiles"][floor_y][1], "ground.grass.top.middle")
        self.assertEqual(visual["tiles"][floor_y][logic["width"] - 1], "ground.grass.top.right")
        result = validate_sockets(visual, manifest)
        self.assertTrue(result["ok"], result)

    def test_validate_sockets_reports_coordinate_direction(self) -> None:
        manifest = self.manifest()
        visual = {
            "width": 2,
            "height": 1,
            "theme": "grass",
            "tiles": [["ground.grass.top.middle", None]],
        }
        result = validate_sockets(visual, manifest)
        self.assertFalse(result["ok"])
        self.assertTrue(any("(0,0) right" in error for error in result["errors"]), result)

    def test_background_sockets_require_fixed_vertical_alignment(self) -> None:
        manifest = self.manifest()
        by_frame = {(tile["sheet"], tile["frame"]): tile for tile in manifest["tiles"]}
        top_sky = [by_frame[("backgrounds", frame)] for frame in (0, 1, 2, 3)]
        middle_sky = [by_frame[("backgrounds", frame)] for frame in (8, 9, 10, 11)]
        bottom_sky = [by_frame[("backgrounds", frame)] for frame in (16, 17, 18, 19)]
        first_middle = middle_sky[0]
        last_middle = middle_sky[-1]
        self.assertIn("air", first_middle["sockets"]["left"])
        self.assertIn("air", last_middle["sockets"]["right"])
        self.assertNotIn("air", first_middle["sockets"]["bottom"])
        self.assertIn("air", first_middle["socket_blacklist"]["bottom"])
        invalid = {"width": 4, "height": 1, "theme": "grass", "tiles": [[tile["id"] for tile in middle_sky]]}
        valid = {
            "width": 4,
            "height": 3,
            "theme": "grass",
            "tiles": [[tile["id"] for tile in top_sky], [tile["id"] for tile in middle_sky], [tile["id"] for tile in bottom_sky]],
        }
        self.assertFalse(validate_sockets(invalid, manifest)["ok"])
        self.assertTrue(validate_sockets(valid, manifest)["ok"])

    def test_auto_conditions_keep_manual_semantic_socket_overrides(self) -> None:
        manifest = self.manifest()
        by_id = {tile["id"]: tile for tile in manifest["tiles"]}
        background = by_id["sky.cloud.left"]
        pipe_body = by_id["pipe.blue.body"]
        actor = by_id["enemy.flying"]
        self.assertEqual(background["condition_override"], "background_vertical_band")
        self.assertEqual(pipe_body["condition_override"], "blue_pipe_stack")
        self.assertEqual(actor["condition_override"], "actor_or_event_air")
        self.assertIn("bg:sky:scene_to_deep:c0", background["sockets"]["bottom"])
        self.assertIn("blue_pipe", pipe_body["sockets"]["top"])
        self.assertEqual(actor["sockets"]["top"], ["air"])

    def test_auto_conditions_use_stable_edge_and_same_color_sockets(self) -> None:
        manifest = self.manifest()
        by_id = {tile["id"]: tile for tile in manifest["tiles"]}
        fill = by_id["ground.dirt.fill"]
        bottom_sockets = fill["sockets"]["bottom"]
        self.assertIn("dirt", bottom_sockets)
        self.assertIn("air", bottom_sockets)
        self.assertTrue(any(socket.startswith("edge:") for socket in fill["condition"]["bottom"]["sockets"]), fill["condition"]["bottom"])
        self.assertTrue(any(socket.startswith("color:") for socket in fill["condition"]["bottom"]["sockets"]), fill["condition"]["bottom"])
        self.assertEqual(fill["condition"]["top"]["connect_policy"], "air_only")
        self.assertEqual(fill["condition"]["top"]["clearance"], 2)

    def test_water_or_pipe_sockets_connect_horizontally_and_vertically(self) -> None:
        manifest = self.manifest()
        by_frame = {(tile["sheet"], tile["frame"]): tile for tile in manifest["tiles"]}
        top = [by_frame[("terrain", frame)] for frame in (33, 34, 35)]
        body = [by_frame[("terrain", frame)] for frame in (53, 54, 55)]
        bottom = [by_frame[("terrain", frame)] for frame in (73, 74, 75)]
        self.assertNotEqual(top[0]["sockets"], {"top": ["air"], "right": ["air"], "bottom": ["air"], "left": ["air"]})
        self.assertIn("air", top[0]["sockets"]["top"])
        self.assertNotIn("air", body[0]["sockets"]["top"])
        valid = {
            "width": 3,
            "height": 3,
            "theme": "grass",
            "tiles": [[tile["id"] for tile in top], [tile["id"] for tile in body], [tile["id"] for tile in bottom]],
        }
        invalid = {"width": 3, "height": 1, "theme": "grass", "tiles": [[tile["id"] for tile in body]]}
        self.assertTrue(validate_sockets(valid, manifest)["ok"])
        self.assertFalse(validate_sockets(invalid, manifest)["ok"])

    def test_blue_pipe_tiles_are_not_random_ground_decorations(self) -> None:
        manifest = self.manifest()
        pipe_tiles = [tile for tile in manifest["tiles"] if tile["role"] in {"pipe_cap_top", "pipe_body", "pipe_cap_bottom"}]
        self.assertTrue(pipe_tiles)
        self.assertTrue(all(tile["placement"]["anchor"] == "vertical_pipe_stack" for tile in pipe_tiles))

    def test_random_ground_decoration_candidates_are_foliage_only(self) -> None:
        manifest = self.manifest()
        random_ground = [
            tile for tile in manifest["tiles"]
            if tile.get("placement", {}).get("anchor") == "ground_top" and tile.get("placement", {}).get("depth") == "front"
        ]
        self.assertTrue(random_ground)
        self.assertTrue(all(tile["role"] == "foliage" for tile in random_ground), random_ground[:5])

    def test_collision_layer_merges_solids_and_keeps_one_way_platforms(self) -> None:
        manifest = self.manifest()
        logic = generate_logic_map({"width": 20, "height": 10, "difficulty": "easy", "seed": 2})
        visual = plan_tiles(logic, manifest, "grass")
        collision = generate_collision_layer(visual, manifest)
        objects = collision["objects"]
        self.assertTrue(any(obj["type"] == "solid" and obj["width"] > 18 for obj in objects), objects)
        self.assertTrue(any(obj["type"] == "one_way" for obj in objects), objects)

    def test_emit_runtime_level_contains_phaser4_physics_profile(self) -> None:
        manifest = self.manifest()
        logic = generate_logic_map({"width": 20, "height": 10, "difficulty": "easy", "seed": 2})
        visual = plan_tiles(logic, manifest, "grass")
        collision = generate_collision_layer(visual, manifest)
        level = emit_runtime_level(logic, visual, collision, manifest)
        self.assertEqual(level["physics_profile"]["engine"], "phaser4_arcade")
        self.assertEqual(level["physics_profile"]["gravity_y"], 900)
        self.assertEqual(level["physics_profile"]["max_jump_height_tiles"], 4)
        self.assertEqual(level["layers"]["visual"], visual["tiles"])
        self.assertTrue(level["layers"]["collision"])
        self.assertTrue(any(event["type"] == "goal" for event in level["layers"]["events"]))


class FrontNoteTests(unittest.TestCase):
    def test_sanitize_front_note_html_removes_script_and_event_handlers(self) -> None:
        html = sanitize_front_note_html('<p onclick="bad()">Hi</p><script>alert(1)</script><a href="javascript:bad()">x</a>')
        self.assertIn("<p>Hi</p>", html)
        self.assertNotIn("script", html.lower())
        self.assertNotIn("onclick", html.lower())
        self.assertNotIn("javascript:", html.lower())

    def test_front_note_html_to_text_preserves_basic_line_breaks(self) -> None:
        self.assertEqual(front_note_html_to_text("<p>第一行</p><div>第二行<br>第三行</div>"), "第一行\n第二行\n第三行")


class TurnTimingTests(unittest.TestCase):
    def test_registered_tool_count_counts_suggested_calls_not_tool_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "test-session")
            payload = {
                "domains": [
                    {
                        "domain": "daily",
                        "suggested_actions": [
                            {"tool_call": 'daily_action(action="weather", target="上海", args={})'},
                            {"tool_call": 'daily_action(action="reminder_create", target="明天下午去上海", args={})'},
                        ],
                    }
                ]
            }
            self.assertEqual(store._registered_tool_count_from_domain_probe(payload, {}), 2)

    def test_turn_timings_are_grouped_in_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "test-session")
            timing_id = store.start_turn_timing("turn-1", "asr", "ASR", {"model": "whisper"})
            store.end_turn_timing(timing_id, metadata={"chars": 4})
            store.record_turn_timing("turn-1", "tool_call", "web_search", duration_seconds=1.25)
            pipeline = store.recent_pipeline(limit=20)
        turns = pipeline["turns"]
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["turn_id"], "turn-1")
        timings = {item["metadata"]["stage"]: item for item in turns[0]["timings"]}
        self.assertEqual(set(timings), {"asr", "tool_call"})
        self.assertEqual(timings["tool_call"]["metadata"]["duration_seconds"], 1.25)

    def test_live_session_listing_pipeline_and_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "current")
            store.add_event("transcript", role="user", lane="text_input", content="当前问题", metadata={"turn_id": "turn-current"})
            store.add_live_note("历史备注", session_id="history")
            sessions = store.list_sessions(limit=10)
            pipeline = store.pipeline_for_session("history", limit=20)

        self.assertEqual(sessions[0]["session_id"], "history")
        self.assertEqual({item["session_id"] for item in sessions}, {"current", "history"})
        self.assertEqual(pipeline["session_id"], "history")
        self.assertEqual(pipeline["items"][-1]["kind"], "live_note")
        self.assertEqual(pipeline["items"][-1]["content"], "历史备注")

    def test_front_note_api_show_is_not_logged_as_tool_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "current")
            store.update_front_note(action="show", tab="live", source="api", allow_empty=True)
            pipeline = store.recent_pipeline(limit=20)
        self.assertFalse([item for item in pipeline["items"] if item["kind"] == "tool_event" and item["content"] == "front_note"])

    def test_front_note_api_show_history_is_hidden_from_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "current")
            store.record_tool_event(
                "front_note",
                {"action": "show", "tab": "live", "source": "api", "content_chars": 0, "media_count": 0},
                {"ok": True, "_summary": "贴好了"},
            )
            pipeline = store.recent_pipeline(limit=20)
        self.assertFalse([item for item in pipeline["items"] if item["kind"] == "tool_event" and item["content"] == "front_note"])

    def test_live_log_source_group_headers_only_change_on_source_switch(self) -> None:
        items = [
            {"kind": "transcript", "role": "user", "metadata": {}},
            {"kind": "live_note", "role": "user", "metadata": {}},
            {"kind": "tool_event", "role": "tool", "metadata": {"ok": True}},
            {"kind": "tool_event", "role": "tool", "metadata": {"ok": False}},
            {"kind": "assistant_reply", "role": "assistant", "metadata": {}},
            {"kind": "timing", "role": "system", "metadata": {"status": "ok"}},
        ]
        groups = [live_log_source_group(item) for item in items]
        headers = [group for index, group in enumerate(groups) if index == 0 or groups[index - 1] != group]
        self.assertEqual(groups, ["user", "user", "agent", "agent", "agent", "agent"])
        self.assertEqual(headers, ["user", "agent"])


class ToolVoiceSummaryTests(unittest.TestCase):
    def test_weather_summary_includes_weather_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "test-session")
            summary = store.tool_voice_summary(
                "get_weather",
                True,
                {
                    "ok": True,
                    "location": "São Paulo",
                    "resolved_location": "Liberdade, Brazil",
                    "temperature_c": "18",
                    "feels_like_c": "18",
                    "humidity_percent": "64",
                    "wind_kmph": "6",
                    "description": "Partly Cloudy",
                },
            )
        self.assertIn("Liberdade, Brazil", summary)
        self.assertIn("18度", summary)
        self.assertIn("Partly Cloudy", summary)

    def test_weather_failure_summary_hides_raw_provider_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "test-session")
            summary = store.tool_voice_summary(
                "get_weather",
                False,
                {
                    "ok": False,
                    "location": "今天怎么样圣保罗",
                    "error": "weather request failed with HTTP 500: location not found: upstream error: opencage: invalid response",
                },
            )
        self.assertEqual(summary, "天气没查到：地点没识别对")
        self.assertNotIn("HTTP", summary)


class SpeechQueueTests(unittest.TestCase):
    def test_tool_start_becomes_obsolete_after_completion(self) -> None:
        speech = SpeechQueue("Tingting", 165, disabled=True)
        self.assertFalse(speech._tool_start_is_obsolete("web_search", "turn-1"))
        speech._mark_tool_speech_completed("web_search", "turn-1")
        self.assertTrue(speech._tool_start_is_obsolete("web_search", "turn-1"))
        self.assertFalse(speech._tool_start_is_obsolete("web_search", "turn-2"))

    def test_recent_speech_gate_only_drops_filler_and_tool(self) -> None:
        speech = SpeechQueue("Tingting", 165, disabled=True)
        original_uniform = speech_module.random.uniform
        try:
            speech_module.random.uniform = lambda _low, _high: 8.0
            speech._mark_speech_triggered()
            with speech._lock:
                self.assertFalse(speech._non_llm_speech_gate_allows_locked("filler", "嗯"))
                self.assertFalse(speech._non_llm_speech_gate_allows_locked("tool", "我查查"))
                self.assertTrue(speech._non_llm_speech_gate_allows_locked("speech", "结果"))
                speech._last_speech_triggered_at -= 12.0
                self.assertTrue(speech._non_llm_speech_gate_allows_locked("filler", "嗯"))
                self.assertTrue(speech._non_llm_speech_gate_allows_locked("tool", "我查查"))
        finally:
            speech_module.random.uniform = original_uniform


class ProLaneFollowupTests(unittest.TestCase):
    def test_pro_followup_speaks_prompt_directly_without_fast_rewrite(self) -> None:
        class FailingFastAgent:
            def respond(self, *_args, **_kwargs) -> str:
                raise AssertionError("pro followup should not call fast lane")

        class FakeSpeech:
            def __init__(self) -> None:
                self.spoken: list[dict[str, object]] = []
                self.filler_stages: list[str] = []
                self.stopped = 0

            def current_generation(self) -> int:
                return 1

            def start_filler_loop(self, stage: str, initial_delay: float = 0.0, interval_range: tuple[float, float] = (0.0, 0.0)) -> threading.Event:
                self.filler_stages.append(stage)
                return threading.Event()

            def stop_filler_loop(self) -> None:
                self.stopped += 1

            def speak(
                self,
                text: str,
                interrupt: bool = False,
                generation: int | None = None,
                force_say: bool = False,
                quick_say_fallback: bool = False,
                turn_id: str = "",
            ) -> None:
                self.spoken.append({
                    "text": text,
                    "interrupt": interrupt,
                    "generation": generation,
                    "force_say": force_say,
                    "quick_say_fallback": quick_say_fallback,
                    "turn_id": turn_id,
                })

        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "test-session")
            speech = FakeSpeech()
            worker = ProLaneWorker(
                SimpleNamespace(
                    back="oss",
                    front="free",
                    followup_interrupt_priority=1,
                    followup_speak_priority=1,
                    followup_dedupe_seconds=120.0,
                    filler_min_interval=8.0,
                    filler_max_interval=15.0,
                ),
                store,
                None,
                FailingFastAgent(),
                speech,
            )
            prompt = "圣保罗目前天气为21摄氏度，晴天，湿度53%，风速9公里每小时。"
            store.trigger_fast_followup(prompt, priority=1)

            self.assertTrue(worker._drain_followups(1, turn_id="turn-1"))
            self.assertEqual(speech.spoken, [{"text": prompt, "interrupt": False, "generation": 1, "force_say": False, "quick_say_fallback": True, "turn_id": "turn-1"}])
            self.assertEqual(speech.filler_stages, ["transition"])
            replies = [event for event in store.recent_pipeline(limit=20)["items"] if event["kind"] == "assistant_reply"]
            self.assertEqual(replies[0]["content"], prompt)
            self.assertEqual(replies[0]["lane"], "oss")
            self.assertEqual(replies[0]["metadata"]["reason"], "pro_followup_direct")


class VoiceBotCancelTests(unittest.TestCase):
    def test_short_recording_cancel_requires_audible_tap(self) -> None:
        bot = VoiceBot.__new__(VoiceBot)
        bot.args = SimpleNamespace(min_record_rms=0.003, min_record_peak=0.015)
        self.assertTrue(bot._short_recording_looks_like_cancel({"rms": 0.01, "peak": 0.05}))
        self.assertFalse(bot._short_recording_looks_like_cancel({"rms": 0.001, "peak": 0.05}))
        self.assertFalse(bot._short_recording_looks_like_cancel({"rms": 0.01, "peak": 0.005}))


class HoldKeyTapTests(unittest.TestCase):
    def test_double_tap_triggers_text_input_without_hold_recording(self) -> None:
        calls: list[tuple[str, str]] = []
        done = threading.Event()
        tap = HoldKeyTap(
            lambda mode: calls.append(("press", mode)),
            lambda mode: calls.append(("release", mode)),
            lambda mode: calls.append(("cancel", mode)) or False,
            lambda mode: (calls.append(("double", mode)), done.set()),
            hold_start_delay=10.0,
            double_click_window=0.4,
        )
        tap.handle_key_down("quality", now=1.0)
        tap.handle_key_up("quality", now=1.05)
        tap.handle_key_down("quality", now=1.2)
        self.assertTrue(done.wait(0.5))
        tap.handle_key_up("quality", now=1.25)
        time.sleep(0.02)
        self.assertIn(("double", "quality"), calls)
        self.assertNotIn(("press", "quality"), calls)

    def test_hold_starts_recording_and_release_stops(self) -> None:
        calls: list[tuple[str, str]] = []
        press_done = threading.Event()
        release_done = threading.Event()
        tap = HoldKeyTap(
            lambda mode: (calls.append(("press", mode)), press_done.set()),
            lambda mode: (calls.append(("release", mode)), release_done.set()),
            lambda mode: calls.append(("cancel", mode)) or True,
            lambda mode: calls.append(("double", mode)),
            hold_start_delay=0.01,
        )
        tap.handle_key_down("simple", now=2.0)
        self.assertTrue(press_done.wait(0.5))
        tap.handle_key_up("simple", now=2.2)
        self.assertTrue(release_done.wait(0.5))
        self.assertEqual(calls[:2], [("press", "simple"), ("release", "simple")])


class DomainProbeTests(unittest.TestCase):
    def test_execution_plan_uses_all_domain_probe_suggestions(self) -> None:
        payload = {
            "domains": [
                {
                    "domain": "daily",
                    "confidence": 0.93,
                    "suggested_actions": [
                        {"tool_call": 'daily_action(action="weather", target="北京", args={"time": "今天"})'},
                        {"tool_call": 'daily_action(action="reminder_create", target="明天下午去北京", args={"time": "明天", "content": "明天下午去北京"})'},
                    ],
                }
            ]
        }
        plan = execution_plan_from_domain_probe(payload)
        self.assertIsNotNone(plan)
        steps = plan["steps"]
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["arguments"]["daily_action"]["action"], "weather")
        self.assertEqual(steps[1]["arguments"]["daily_action"]["action"], "reminder_create")

    def test_execution_plan_includes_computer_action_from_probe(self) -> None:
        payload = probe_domains("打开 reminder，提醒我两个小时后出门")
        plan = execution_plan_from_domain_probe(payload)
        self.assertIsNotNone(plan)
        steps = plan["steps"]
        self.assertEqual(steps[0]["suggested_tools"], ["computer_action"])
        self.assertEqual(steps[0]["arguments"]["computer_action"]["target"], "Reminders")
        self.assertTrue(any(step["suggested_tools"] == ["computer_action"] and step["arguments"]["computer_action"]["target"] == "Reminders" for step in steps))
        self.assertTrue(any(step["suggested_tools"] == ["daily_action"] and step["arguments"]["daily_action"]["action"] == "reminder_create" for step in steps))

    def test_app_open_prefix_does_not_pollute_daily_weather_probe(self) -> None:
        original = domain_probe._current_address_for_probe
        domain_probe._current_address_for_probe = lambda: {"ok": True, "address": "Liberdade，São Paulo，Brazil"}
        try:
            payload = probe_domains("打开reminder,今天天气怎么样?提醒我两小时之后出门")
        finally:
            domain_probe._current_address_for_probe = original
        plan = execution_plan_from_domain_probe(payload)
        self.assertIsNotNone(plan)
        steps = plan["steps"]
        self.assertTrue(any(step["suggested_tools"] == ["computer_action"] and step["arguments"]["computer_action"]["target"] == "Reminders" for step in steps))
        weather_steps = [step for step in steps if step["suggested_tools"] == ["daily_action"] and step["arguments"]["daily_action"]["action"] == "weather"]
        self.assertTrue(weather_steps)
        self.assertIn("Liberdade", weather_steps[0]["arguments"]["daily_action"]["target"])
        reminder_steps = [step for step in steps if step["suggested_tools"] == ["daily_action"] and step["arguments"]["daily_action"]["action"] == "reminder_create"]
        self.assertTrue(reminder_steps)
        self.assertEqual(reminder_steps[0]["arguments"]["daily_action"]["args"]["time"], "两小时之后")

    def test_direct_probe_response_does_not_shortcut_multiple_actions(self) -> None:
        payload = {
            "domains": [
                {
                    "domain": "daily",
                    "confidence": 0.93,
                    "suggested_actions": [
                        {
                            "tool_call": 'daily_action(action="weather", target="上海", args={})',
                            "answer": {"ok": True, "location": "上海", "temperature_c": "24"},
                        },
                        {
                            "tool_call": 'daily_action(action="reminder_create", target="明天下午去上海", args={})',
                        },
                    ],
                }
            ]
        }
        self.assertEqual(direct_response_from_domain_probe(payload), "")

    def test_direct_probe_response_does_not_shortcut_single_answer(self) -> None:
        payload = {
            "domains": [
                {
                    "domain": "daily",
                    "confidence": 0.93,
                    "suggested_actions": [
                        {
                            "tool_call": 'daily_action(action="weather", target="上海", args={})',
                            "answer": {"ok": True, "location": "上海", "temperature_c": "24"},
                        }
                    ],
                }
            ]
        }
        self.assertEqual(direct_response_from_domain_probe(payload), "")

    def test_daily_segmenter_splits_punctuation_before_new_task(self) -> None:
        text = "今天圣保罗,啊不对不对北京啊不对上海天气怎么样, 提醒我明天下午去上海"
        self.assertEqual(
            [item["text"] for item in split_daily_segments(text)],
            ["今天圣保罗,啊不对不对北京啊不对上海天气怎么样", "提醒我明天下午去上海"],
        )

    def test_daily_segmenter_splits_unpunctuated_reminder_after_weather(self) -> None:
        text = "今天我孙子是明天我孙子是的天气怎么样明天下午提醒我去日本"
        self.assertEqual(
            [item["text"] for item in split_daily_segments(text)],
            ["今天我孙子是明天我孙子是的天气怎么样", "明天下午提醒我去日本"],
        )

    def test_daily_probe_extracts_weather_location_from_spoken_query(self) -> None:
        payload = probe_domains("今天天气怎么样圣保罗")
        domains = {item["domain"]: item for item in payload["domains"]}
        action = domains["daily"]["suggested_actions"][0]
        self.assertIn('daily_action(action="weather"', action["tool_call"])
        self.assertIn('target="圣保罗"', action["tool_call"])
        self.assertEqual(extract_weather_location("今天天气怎么样圣保罗"), "圣保罗")

    def test_computer_probe_resolves_app_alias(self) -> None:
        payload = probe_domains("关掉 inna")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("computer", domains)
        computer = domains["computer"]
        self.assertEqual(computer["intent"], "close_app")
        self.assertIn('computer_action(action="close_app", target="IINA"', computer["suggested_actions"][0]["tool_call"])
        self.assertIn("app close", computer["suggested_actions"][0]["desc"])

    def test_computer_probe_recommends_computer_use_for_gui_keys(self) -> None:
        payload = probe_domains("帮我按一下 Chrome 的 f 快捷键")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("computer", domains)
        action = domains["computer"]["suggested_actions"][0]
        self.assertIn('computer_action(action="computer_use"', action["tool_call"])

    def test_computer_probe_resolves_quicktime_alias(self) -> None:
        payload = probe_domains("关掉 quicktime")
        domains = {item["domain"]: item for item in payload["domains"]}
        action = domains["computer"]["suggested_actions"][0]
        self.assertIn('target="QuickTime Player"', action["tool_call"])

    def test_computer_probe_resolves_reminders_alias(self) -> None:
        payload = probe_domains("打开 reminder")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("computer", domains)
        self.assertNotIn("daily", domains)
        action = domains["computer"]["suggested_actions"][0]
        self.assertIn('computer_action(action="open_app", target="Reminders"', action["tool_call"])

    def test_computer_probe_uses_generated_local_app_alias(self) -> None:
        original = domain_probe._local_app_alias_catalog
        domain_probe._local_app_alias_catalog = lambda: (
            {"app": "pcsuite", "alias": "vivo", "alias_key": "vivo", "source": "installed_app", "bundle_id": "com.vivo.pcsuite", "running": False},
        )
        try:
            payload = probe_domains("关掉 vivo 办公套件")
        finally:
            domain_probe._local_app_alias_catalog = original
        domains = {item["domain"]: item for item in payload["domains"]}
        action = domains["computer"]["suggested_actions"][0]
        self.assertIn('target="pcsuite"', action["tool_call"])

    def test_computer_probe_prefers_registered_program_over_running_app_fuzzy(self) -> None:
        original = domain_probe._local_app_alias_catalog
        domain_probe._local_app_alias_catalog = lambda: (
            {"app": "sociallayerd", "alias": "sociallayerd", "alias_key": "sociallayerd", "source": "running_app", "bundle_id": "com.apple.sociallayerd", "running": True},
        )
        try:
            payload = probe_domains(
                "启动Flappy Bird",
                context={
                    "registered_programs": [
                        {
                            "workspace_id": "flappy-1",
                            "path": "/tmp/flappy",
                            "title": "Flappy Bird",
                            "aliases": ["flappy bird", "flappybird"],
                            "program": {"name": "Flappy Bird", "aliases": ["Flappy Bird", "flappybird"], "status": "ready"},
                        }
                    ]
                },
            )
        finally:
            domain_probe._local_app_alias_catalog = original
        computer = {item["domain"]: item for item in payload["domains"]}["computer"]
        action = computer["suggested_actions"][0]
        self.assertIn('computer_action(action="open_program", target="Flappy Bird"', action["tool_call"])
        self.assertNotIn("sociallayerd", json.dumps(computer, ensure_ascii=False))

    def test_computer_probe_suggests_multiple_actions_for_file_display_task(self) -> None:
        payload = probe_domains("download 文件里打开 [億次研同好會&三明治擺爛組] 日本三國 NipponSangoku [09][1080P][繁日內嵌]，然后移到外接屏幕")
        domains = {item["domain"]: item for item in payload["domains"]}
        computer = domains["computer"]
        tool_calls = [item["tool_call"] for item in computer["suggested_actions"]]
        self.assertEqual(computer["intent"], "open_file_then_move_window")
        self.assertTrue(any('action="open_file_and_move_to_display"' in call for call in tool_calls))
        self.assertTrue(any('action="open_file"' in call for call in tool_calls))
        self.assertTrue(any('action="move_window_to_display"' in call for call in tool_calls))
        self.assertTrue(any('"folder": "~/Downloads"' in call and '"display": "external"' in call for call in tool_calls))

    def test_daily_probe_routes_note_to_live_front_note(self) -> None:
        payload = probe_domains("在便签写下明天开会")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("daily", domains)
        action = domains["daily"]["suggested_actions"][0]
        self.assertIn('daily_action(action="note_live"', action["tool_call"])
        self.assertIn('target="明天开会"', action["tool_call"])

    def test_note_capture_with_deictic_news_does_not_route_to_search(self) -> None:
        payload = probe_domains("记一下这件新闻")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertNotIn("search", domains)
        action = domains["daily"]["suggested_actions"][0]
        self.assertIn('daily_action(action="note_live"', action["tool_call"])

    def test_explicit_long_term_memory_still_routes_to_context_note(self) -> None:
        payload = probe_domains("帮我记一下我喜欢圣保罗")
        domains = {item["domain"]: item for item in payload["domains"]}
        action = domains["daily"]["suggested_actions"][0]
        self.assertIn('daily_action(action="memory"', action["tool_call"])

    def test_daily_probe_routes_time_map_and_reminder_to_fat_tool(self) -> None:
        cases = [
            ("现在几点", "time"),
            ("去 Paulista 怎么走", "map"),
            ("提醒我明天买咖啡", "reminder_create"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                payload = probe_domains(text)
                action = {item["domain"]: item for item in payload["domains"]}["daily"]["suggested_actions"][0]
                self.assertIn(f'daily_action(action="{expected}"', action["tool_call"])

    def test_daily_probe_uses_semantic_slot_parser_result(self) -> None:
        original = domain_probe.parse_daily_actions
        domain_probe.parse_daily_actions = lambda text: [{
            "action": "map",
            "target": "Paulista",
            "args": {"time": "", "content": "", "modifiers": []},
            "spans": [{"text": "Paulista", "type": "TARGET"}],
            "resolved": {"action": "map", "target": "Paulista"},
            "segment": {"text": text, "start": 0, "end": len(text), "index": 0},
        }]
        try:
            payload = probe_domains("去Paulista怎么走")
        finally:
            domain_probe.parse_daily_actions = original
        daily = {item["domain"]: item for item in payload["domains"]}["daily"]
        action = daily["suggested_actions"][0]
        self.assertIn('daily_action(action="map", target="Paulista"', action["tool_call"])
        self.assertIn('"mode": "route"', action["tool_call"])
        self.assertIn("semantic slot parser", action["desc"])
        self.assertEqual(daily["matched_entities"][0]["source"], "semantic_slot_parser")

    def test_daily_probe_semantic_weather_without_target_uses_current_address(self) -> None:
        original_parser = domain_probe.parse_daily_actions
        original_address = domain_probe._current_address_for_probe
        domain_probe.parse_daily_actions = lambda text: [{
            "action": "weather",
            "target": "",
            "args": {"time": "今天", "content": "", "modifiers": []},
            "spans": [{"text": "天气", "type": "ACTION"}],
            "resolved": {"action": "weather"},
            "segment": {"text": text, "start": 0, "end": len(text), "index": 0},
        }]
        domain_probe._current_address_for_probe = lambda: {"ok": True, "address": "Liberdade，São Paulo，Brazil"}
        try:
            payload = probe_domains("今天天气怎么样")
        finally:
            domain_probe.parse_daily_actions = original_parser
            domain_probe._current_address_for_probe = original_address
        action = {item["domain"]: item for item in payload["domains"]}["daily"]["suggested_actions"][0]
        self.assertIn('daily_action(action="weather"', action["tool_call"])
        self.assertIn('target="Liberdade，São Paulo，Brazil"', action["tool_call"])
        self.assertIn('"location_source": "current_address"', action["tool_call"])

    def test_daily_probe_semantic_cancel_does_not_suggest_action(self) -> None:
        original = domain_probe.parse_daily_actions
        domain_probe.parse_daily_actions = lambda text: [{
            "action": "weather",
            "target": "北京",
            "args": {"time": "今天", "content": "", "modifiers": []},
            "spans": [{"text": "算了", "type": "NEGATION"}],
            "resolved": {"action": "weather", "target": "北京", "cancelled": True},
            "cancelled": True,
            "segment": {"text": text, "start": 0, "end": len(text), "index": 0},
        }]
        try:
            payload = probe_domains("今天北京天气算了")
        finally:
            domain_probe.parse_daily_actions = original
        daily = {item["domain"]: item for item in payload["domains"]}["daily"]
        self.assertEqual(daily["intent"], "cancelled")
        self.assertEqual(daily["suggested_actions"], [])

    def test_daily_probe_semantic_multi_segment_suggests_multiple_actions(self) -> None:
        original = domain_probe.parse_daily_actions
        domain_probe.parse_daily_actions = lambda text: [
            {
                "action": "weather",
                "target": "上海",
                "args": {"time": "今天", "content": "", "modifiers": []},
                "spans": [{"text": "上海", "type": "TARGET"}, {"text": "天气怎么样", "type": "ACTION"}],
                "resolved": {"action": "weather", "target": "上海"},
                "segment": {"text": "今天上海天气怎么样", "start": 0, "end": 9, "index": 0},
            },
            {
                "action": "reminder_create",
                "target": "明天下午去上海",
                "args": {"time": "明天", "content": "下午去上海", "modifiers": []},
                "spans": [{"text": "提醒我", "type": "ACTION"}, {"text": "明天下午去上海", "type": "CONTENT"}],
                "resolved": {"action": "reminder_create", "target": "明天下午去上海"},
                "segment": {"text": "记得提醒我明天下午去上海", "start": 10, "end": 24, "index": 1},
            },
        ]
        try:
            payload = probe_domains("今天上海天气怎么样，然后记得提醒我明天下午去上海")
        finally:
            domain_probe.parse_daily_actions = original
        daily = {item["domain"]: item for item in payload["domains"]}["daily"]
        calls = [item["tool_call"] for item in daily["suggested_actions"]]
        self.assertTrue(any('daily_action(action="weather", target="上海"' in call for call in calls))
        self.assertTrue(any('daily_action(action="reminder_create", target="明天下午去上海"' in call for call in calls))
        self.assertEqual(daily["matched_entities"][0]["type"], "semantic_segments")

    def test_daily_probe_weather_without_place_injects_current_location_probe(self) -> None:
        original = domain_probe._current_address_for_probe
        domain_probe._current_address_for_probe = lambda: {"ok": True, "address": "Liberdade，São Paulo，Brazil"}
        try:
            payload = probe_domains("今天天气怎么样")
        finally:
            domain_probe._current_address_for_probe = original
        action = {item["domain"]: item for item in payload["domains"]}["daily"]["suggested_actions"][0]
        self.assertIn('daily_action(action="weather"', action["tool_call"])
        self.assertIn('target="Liberdade，São Paulo，Brazil"', action["tool_call"])
        self.assertIn('"location_source": "current_address"', action["tool_call"])
        self.assertNotIn("use_current_location", action["tool_call"])

    def test_daily_probe_rain_without_place_does_not_use_action_as_location(self) -> None:
        original = domain_probe._current_address_for_probe
        domain_probe._current_address_for_probe = lambda: {"ok": True, "address": "Liberdade，São Paulo，Brazil"}
        try:
            payload = probe_domains("明天下雨吗")
        finally:
            domain_probe._current_address_for_probe = original
        action = {item["domain"]: item for item in payload["domains"]}["daily"]["suggested_actions"][0]
        self.assertIn('daily_action(action="weather"', action["tool_call"])
        self.assertIn('target="Liberdade，São Paulo，Brazil"', action["tool_call"])
        self.assertNotIn('target="下雨吗"', action["tool_call"])

    def test_daily_probe_rain_with_le_without_place_does_not_use_action_as_location(self) -> None:
        original = domain_probe._current_address_for_probe
        domain_probe._current_address_for_probe = lambda: {"ok": True, "address": "Liberdade，São Paulo，Brazil"}
        try:
            payload = probe_domains("明天下雨了吗")
        finally:
            domain_probe._current_address_for_probe = original
        action = {item["domain"]: item for item in payload["domains"]}["daily"]["suggested_actions"][0]
        self.assertIn('target="Liberdade，São Paulo，Brazil"', action["tool_call"])
        self.assertNotIn("下雨了吗", action["tool_call"])

    def test_daily_probe_traditional_rain_without_place_uses_current_location(self) -> None:
        original = domain_probe._current_address_for_probe
        domain_probe._current_address_for_probe = lambda: {"ok": True, "address": "Liberdade，São Paulo，Brazil"}
        try:
            payload = probe_domains("明天下雨嗎")
        finally:
            domain_probe._current_address_for_probe = original
        action = {item["domain"]: item for item in payload["domains"]}["daily"]["suggested_actions"][0]
        self.assertIn('target="Liberdade，São Paulo，Brazil"', action["tool_call"])
        self.assertNotIn("下雨嗎", action["tool_call"])

    def test_weather_query_candidates_degrade_full_address_to_city(self) -> None:
        candidates = domain_probe._weather_query_candidates("Avenida Engenheiro Luís Carlos Berrini, 901，São Paulo，SP，Brazil")
        self.assertIn("São Paulo, SP, Brazil", candidates)
        self.assertIn("São Paulo, Brazil", candidates)
        self.assertIn("São Paulo", candidates)
        self.assertNotIn("901", candidates)

    def test_daily_probe_generalizes_weather_intents(self) -> None:
        original = domain_probe._current_address_for_probe
        domain_probe._current_address_for_probe = lambda: {"ok": True, "address": "Liberdade，São Paulo，Brazil"}
        try:
            for text in ["今天温度怎么样", "现在几度", "这边下雨吗", "今天要不要带伞"]:
                with self.subTest(text=text):
                    payload = probe_domains(text)
                    action = {item["domain"]: item for item in payload["domains"]}["daily"]["suggested_actions"][0]
                    self.assertIn('daily_action(action="weather"', action["tool_call"])
                    self.assertIn('target="Liberdade，São Paulo，Brazil"', action["tool_call"])
        finally:
            domain_probe._current_address_for_probe = original

    def test_daily_probe_weather_location_cleanup_handles_colloquial_words(self) -> None:
        self.assertEqual(domain_probe.weather_location_from_text("圣保罗现在几度"), "圣保罗")
        self.assertEqual(domain_probe.weather_location_from_text("上海今天下雨吗"), "上海")
        self.assertEqual(domain_probe.weather_location_from_text("上海天气怎么样明天下午提醒我去上海"), "上海")

    def test_english_daily_probe_routes_weather_time_map_and_reminder(self) -> None:
        cases = [
            ("what's the weather in São Paulo today", "weather", "São Paulo"),
            ("what time is it", "time", ""),
            ("directions to Paulista Avenue", "map", "Paulista Avenue"),
            ("remind me to buy coffee tomorrow", "reminder_create", "buy coffee tomorrow"),
        ]
        for text, action_name, target in cases:
            with self.subTest(text=text):
                payload = probe_domains(text)
                domains = {item["domain"]: item for item in payload["domains"]}
                self.assertIn("daily", domains)
                action = domains["daily"]["suggested_actions"][0]
                self.assertIn(f'daily_action(action="{action_name}"', action["tool_call"])
                if target:
                    self.assertIn(target, action["tool_call"])

    def test_english_weather_without_place_does_not_use_grammar_as_location(self) -> None:
        original = domain_probe._current_address_for_probe
        domain_probe._current_address_for_probe = lambda: {"ok": True, "address": "Liberdade，São Paulo，Brazil"}
        try:
            payload = probe_domains("what's the weather")
        finally:
            domain_probe._current_address_for_probe = original
        action = {item["domain"]: item for item in payload["domains"]}["daily"]["suggested_actions"][0]
        self.assertIn('daily_action(action="weather"', action["tool_call"])
        self.assertIn('target="Liberdade，São Paulo，Brazil"', action["tool_call"])
        self.assertNotIn("what", action["tool_call"].lower())

    def test_english_computer_and_coding_probe(self) -> None:
        computer_payload = probe_domains("open Chrome")
        computer_domains = {item["domain"]: item for item in computer_payload["domains"]}
        self.assertIn("computer", computer_domains)
        computer_action = computer_domains["computer"]["suggested_actions"][0]["tool_call"]
        self.assertIn('computer_action(action="open_app"', computer_action)
        self.assertIn("Google Chrome", computer_action)

        coding_payload = probe_domains("create a pacman game")
        coding_domains = {item["domain"]: item for item in coding_payload["domains"]}
        self.assertIn("computer", coding_domains)
        action = coding_domains["computer"]["suggested_actions"][0]
        self.assertIn('computer_action(action="develop_app"', action["tool_call"])
        self.assertIn("pacman game", action["tool_call"])

    def test_daily_probe_keeps_asr_location_surface_without_weather_override(self) -> None:
        original = domain_probe.parse_daily_actions
        domain_probe.parse_daily_actions = lambda text: [{
            "action": "weather",
            "target": "我孙子试",
            "args": {"time": "今天", "content": "", "modifiers": []},
            "spans": [{"text": "我孙子试", "type": "TARGET"}, {"text": "天气怎么样", "type": "ACTION"}],
            "resolved": {"daily_action_call": {"action": "weather", "target": "我孙子试", "args": {}}},
        }]
        try:
            payload = probe_domains("今天我孙子试天气怎么样")
        finally:
            domain_probe.parse_daily_actions = original
        action = {item["domain"]: item for item in payload["domains"]}["daily"]["suggested_actions"][0]
        self.assertIn('daily_action(action="weather", target="我孙子试"', action["tool_call"])
        self.assertNotIn("我孙子市", action["tool_call"])

    def test_split_daily_segments_handles_weather_then_reminder(self) -> None:
        segments = split_daily_segments("今天圣保罗，不对北京，不对上海天气怎么样，然后提醒我明天下午去上海")
        self.assertEqual([item["text"] for item in segments], ["今天圣保罗，不对北京，不对上海天气怎么样", "提醒我明天下午去上海"])

    def test_daily_segments_keep_video_open_out_of_weather_target(self) -> None:
        segments = split_daily_segments("打开reminder,打开搞笑猫猫视频。明天北京天气怎么样?提醒我明天去北京")
        self.assertEqual(
            [item["text"] for item in segments],
            ["打开reminder", "打开搞笑猫猫视频", "明天北京天气怎么样", "提醒我明天去北京"],
        )
        payload = probe_domains("打开reminder,打开搞笑猫猫视频。明天北京天气怎么样?提醒我明天去北京")
        daily = {item["domain"]: item for item in payload["domains"]}.get("daily")
        self.assertIsNotNone(daily)
        calls = [item["tool_call"] for item in daily["suggested_actions"]]
        self.assertTrue(any('daily_action(action="weather", target="北京"' in call for call in calls))
        self.assertTrue(any('daily_action(action="reminder_create", target="明天去北京"' in call for call in calls))
        self.assertFalse(any("搞笑猫猫视频北京" in call for call in calls))

    def test_daily_parser_reminder_time_uses_original_text(self) -> None:
        actions = parse_daily_actions("提醒我明天下午去上海")
        self.assertEqual(actions[0]["action"], "reminder_create")
        self.assertEqual(actions[0]["target"], "明天下午去上海")
        self.assertEqual(actions[0]["args"]["time"], "明天下午")

    def test_daily_parser_splits_multiple_reminders_with_inherited_intent(self) -> None:
        actions = parse_daily_actions("提醒我明天下午去北京,后天八点去山东,然后今天下午得去买点东西")
        self.assertEqual([item["action"] for item in actions], ["reminder_create", "reminder_create", "reminder_create"])
        self.assertEqual([item["target"] for item in actions], ["明天下午去北京", "后天八点去山东", "今天下午得去买点东西"])
        self.assertEqual([item["args"]["time"] for item in actions], ["明天下午", "后天八点", "今天下午"])
        payload = probe_domains("提醒我明天下午去北京,后天八点去山东,然后今天下午得去买点东西")
        daily = {item["domain"]: item for item in payload["domains"]}.get("daily")
        self.assertIsNotNone(daily)
        calls = [item["tool_call"] for item in daily["suggested_actions"]]
        self.assertEqual(sum('daily_action(action="reminder_create"' in call for call in calls), 3)
        self.assertTrue(any('target="明天下午去北京"' in call for call in calls))
        self.assertTrue(any('target="后天八点去山东"' in call for call in calls))
        self.assertTrue(any('target="今天下午得去买点东西"' in call for call in calls))

    def test_daily_parser_reminder_relative_duration_uses_original_text(self) -> None:
        actions = parse_daily_actions("提醒我15分钟后喝水")
        self.assertEqual(actions[0]["action"], "reminder_create")
        self.assertEqual(actions[0]["target"], "15分钟后喝水")
        self.assertEqual(actions[0]["args"]["time"], "15分钟后")
        actions = parse_daily_actions("两个小时后提醒我出门")
        self.assertEqual(actions[0]["action"], "reminder_create")
        self.assertEqual(actions[0]["target"], "两个小时后出门")
        self.assertEqual(actions[0]["args"]["time"], "两个小时后")
        actions = parse_daily_actions("提醒我两小时之后出门")
        self.assertEqual(actions[0]["args"]["time"], "两小时之后")

    def test_daily_parser_deictic_weather_uses_current_location_later(self) -> None:
        actions = parse_daily_actions("这边下雨吗")
        self.assertEqual(actions[0]["action"], "weather")
        self.assertEqual(actions[0]["target"], "")

    def test_domain_probe_prompt_contains_priority_rule(self) -> None:
        prompt = format_domain_probe_prompt(probe_domains("查查特朗普最近干了什么"))
        self.assertIn("Domain probe JSON", prompt)
        self.assertIn("confidence >= 0.8", prompt)

    def test_domain_probe_prompt_omits_context_and_matched_entities(self) -> None:
        payload = {
            "input": "去Paulista怎么走",
            "nearby_session": {
                "summary": "用户刚才在问路线",
                "recent_events": [{"kind": "transcript", "role": "user", "text": "上一轮输入"}],
            },
            "domains": [
                {
                    "domain": "daily",
                    "confidence": 0.93,
                    "intent": "map",
                    "context": "debug-only context",
                    "matched_entities": [{"type": "semantic_segments", "value": [{"large": "payload"}]}],
                    "suggested_actions": [
                        {
                            "tool_call": 'daily_action(action="map", target="Paulista", args={"mode":"route"})',
                            "desc": "route matched",
                            "confidence": 0.93,
                            "answer": {"ok": True, "web_url": "https://maps.apple.com/?daddr=Paulista"},
                        }
                    ],
                }
            ],
            "available_domains": {"daily": "debug-only descriptions"},
        }
        prompt = format_domain_probe_prompt(payload)
        self.assertIn("daily_action", prompt)
        self.assertIn("route matched", prompt)
        self.assertIn("nearby_session", prompt)
        self.assertIn("用户刚才在问路线", prompt)
        self.assertNotIn("matched_entities", prompt)
        self.assertNotIn("debug-only context", prompt)
        self.assertNotIn("available_domains", prompt)

    def test_domain_probe_does_not_attach_failed_prefetch_answer(self) -> None:
        action = domain_probe._suggested(
            "weather",
            {"target": "", "args": {}},
            0.93,
            "weather without location",
            tool="daily_action",
        )[0]
        self.assertIn("tool_call", action)
        self.assertNotIn("answer", action)

    def test_domain_probe_context_is_compacted_into_prompt(self) -> None:
        payload = probe_domains(
            "今天天气怎么样",
            context={
                "summary": "用户上一轮问过圣保罗。",
                "front_note_context": "常用地点是 São Paulo。",
                "recent_events": [
                    {"kind": "transcript", "role": "user", "content": "刚才说这边"},
                    {"kind": "assistant_reply", "role": "assistant", "content": "我看一下"},
                ],
            },
        )
        prompt = format_domain_probe_prompt(payload)
        self.assertIn("nearby_session", prompt)
        self.assertIn("用户上一轮问过圣保罗", prompt)
        self.assertIn("常用地点是 São Paulo", prompt)

    def test_python_probe_routes_explicit_codex_cli_development(self) -> None:
        payload = probe_domains("用 codex-cli 帮我修这个 bug")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("computer", domains)
        action = domains["computer"]["suggested_actions"][0]
        self.assertIn('computer_action(action="develop_app"', action["tool_call"])
        self.assertIn('"prompt"', action["tool_call"])
        self.assertIn('"executor": "codex"', action["tool_call"])

    def test_python_probe_does_not_route_antigravity_executor_while_disabled(self) -> None:
        payload = probe_domains("用 antigravity 帮我写个小工具")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("computer", domains)
        action = domains["computer"]["suggested_actions"][0]
        self.assertIn('computer_action(action="develop_app"', action["tool_call"])
        self.assertNotIn('"executor": "antigravity"', action["tool_call"])
        self.assertNotIn("antigravity 帮我", action["tool_call"])

    def test_python_probe_routes_general_code_generation_to_coding_action(self) -> None:
        payload = probe_domains("我想玩吃豆人了,给我写个吃豆人游戏")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("computer", domains)
        action = domains["computer"]["suggested_actions"][0]
        self.assertIn('computer_action(action="develop_app"', action["tool_call"])
        self.assertIn("吃豆人", action["tool_call"])

    def test_python_probe_routes_playful_generation_without_code_noun_to_coding_action(self) -> None:
        payload = probe_domains("想玩吃豆仁了,写个吃豆仁玩一下")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("computer", domains)
        action = domains["computer"]["suggested_actions"][0]
        self.assertIn('computer_action(action="develop_app"', action["tool_call"])
        self.assertIn("吃豆仁", action["tool_call"])

    def test_python_probe_routes_screen_animation_generation_to_coding_action(self) -> None:
        payload = probe_domains("写个猪头在屏幕前乱碰")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("computer", domains)
        action = domains["computer"]["suggested_actions"][0]
        self.assertIn('computer_action(action="develop_app"', action["tool_call"])
        self.assertIn("猪头", action["tool_call"])

    def test_python_probe_routes_python_stack_development_to_coding_action(self) -> None:
        payload = probe_domains("用 pywebview 和 uv 做个无边框置顶透明悬浮窗口")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("computer", domains)
        self.assertNotIn("daily", domains)
        action = domains["computer"]["suggested_actions"][0]
        self.assertIn('computer_action(action="develop_app"', action["tool_call"])
        self.assertIn("pywebview", action["tool_call"])

    def test_domain_probe_always_injects_low_confidence_codex_delegate(self) -> None:
        payload = probe_domains("今天天气怎么样")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("daily", domains)
        self.assertIn("computer", domains)
        delegate_calls = [
            action["tool_call"]
            for action in domains["computer"].get("suggested_actions", [])
            if 'action="delegate_to_codex"' in action.get("tool_call", "")
        ]
        self.assertEqual(len(delegate_calls), 1)
        plan = execution_plan_from_domain_probe(payload)
        planned_calls = json.dumps(plan, ensure_ascii=False)
        self.assertNotIn("delegate_to_codex", planned_calls)
        self.assertIn("daily_action", planned_calls)

    def test_tool_payload_uses_probe_subset_not_all_default_tools(self) -> None:
        weather_tools = tool_names_for_domain_probe(probe_domains("今天天气怎么样"))
        self.assertEqual(weather_tools, {"daily_action", "trigger_fast_followup"})

        develop_tools = tool_names_for_domain_probe(probe_domains("开发一个 flappy bird"))
        self.assertEqual(develop_tools, {"computer_action", "trigger_fast_followup"})

        fallback_tools = tool_names_for_domain_probe({"input": "完全不知道怎么做", "domains": []})
        self.assertEqual(fallback_tools, set())

    def test_empty_domain_probe_prompt_still_exposes_codex_delegate(self) -> None:
        prompt = format_domain_probe_prompt({"input": "开发植物大栈僵尸", "domains": []})
        self.assertIn('computer_action(action="delegate_to_codex"', prompt)
        self.assertIn("开发植物大栈僵尸", prompt)
        self.assertIn("直接执行这个兜底", prompt)

    def test_development_phrase_beats_daily_map_probe(self) -> None:
        payload = probe_domains("植物大战僵尸去哪了?开发植物大战僵尸")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("computer", domains)
        self.assertNotIn("daily", domains)
        action = domains["computer"]["suggested_actions"][0]
        self.assertIn('computer_action(action="develop_app"', action["tool_call"])
        self.assertIn("植物大战僵尸", action["tool_call"])

    def test_debug_project_phrase_routes_to_develop_app_not_python_code(self) -> None:
        for text in ["Debug一下植物大转僵尸", "Dbug 植物大战僵尸"]:
            with self.subTest(text=text):
                payload = probe_domains(text)
                domains = {item["domain"]: item for item in payload["domains"]}
                self.assertIn("computer", domains)
                self.assertEqual(domains["computer"]["intent"], "develop_app")
                action = domains["computer"]["suggested_actions"][0]
                self.assertIn('computer_action(action="develop_app"', action["tool_call"])
                self.assertNotIn("run_python", action["tool_call"])
                self.assertTrue(coding_or_debug_task_requested(text))
        self.assertFalse(coding_or_debug_task_requested("运行这个 Python 脚本"))

    def test_python_probe_does_not_route_generic_script_to_codex(self) -> None:
        payload = probe_domains("运行这个 Python 脚本")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertNotIn("python", domains)
        self.assertIn("computer", domains)
        action = domains["computer"]["suggested_actions"][0]
        self.assertIn('computer_action(action="delegate_to_codex"', action["tool_call"])

    def test_video_correction_keeps_negative_constraint(self) -> None:
        query = video_search_query_from_text("Chrome打开YMCA不是打开官方的MTV随便一个就行")
        self.assertIn("YMCA", query)
        self.assertIn("music video", query)
        self.assertIn("-official", query)
        self.assertNotIn("不是", query)

    def test_domain_probe_routes_ymca_mtv_to_search(self) -> None:
        payload = probe_domains("Chrome打开YMCA不是打开官方的MTV随便一个就行")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("search", domains)
        tool_call = domains["search"]["suggested_actions"][0]["tool_call"]
        self.assertIn("YMCA", tool_call)
        self.assertNotIn("不是", tool_call)

    def test_search_and_open_pages_does_not_fallback_to_codex_or_open_program(self) -> None:
        payload = probe_domains("查一查特朗普全身的衣服都什么牌子了,然后把那些网页都打开")
        domains = {item["domain"]: item for item in payload["domains"]}
        self.assertIn("search", domains)
        all_calls = "\n".join(
            action.get("tool_call", "")
            for domain in payload["domains"]
            for action in domain.get("suggested_actions", [])
        )
        self.assertIn("web_search", all_calls)
        self.assertNotIn("delegate_to_codex", all_calls)
        self.assertNotIn("open_program", all_calls)
        selected_tools = tool_names_for_domain_probe(payload)
        self.assertIn("web_search", selected_tools)
        self.assertIn("open_url_in_browser", selected_tools)

    def test_video_search_open_plan_starts_with_search_then_browser_open(self) -> None:
        payload = probe_domains("打开搞笑猫猫视频")
        plan = execution_plan_from_domain_probe(payload)
        self.assertIsNotNone(plan)
        tools = [step["suggested_tools"][0] for step in plan["steps"]]
        self.assertEqual(tools[0], "web_search")
        self.assertIn("open_url_in_browser", tools)
        self.assertLess(tools.index("web_search"), tools.index("open_url_in_browser"))


class WorkspaceLayoutTests(unittest.TestCase):
    def test_auto_layout_three_windows_is_left_main_and_right_stack(self) -> None:
        rects = workspace_layout_rects(3, {"left": 0, "top": 38, "width": 1000, "height": 800}, mode="auto")
        self.assertEqual(
            rects,
            [
                {"x": 0, "y": 38, "width": 500, "height": 800},
                {"x": 500, "y": 38, "width": 500, "height": 400},
                {"x": 500, "y": 438, "width": 500, "height": 400},
            ],
        )

    def test_parallel_layout_splits_evenly(self) -> None:
        rects = workspace_layout_rects(4, {"left": 0, "top": 38, "width": 1000, "height": 800}, mode="parallel")
        self.assertEqual([rect["width"] for rect in rects], [250, 250, 250, 250])
        self.assertEqual([rect["x"] for rect in rects], [0, 250, 500, 750])


class ProGuardTests(unittest.TestCase):
    def test_short_tool_name_strips_provider_prefix(self) -> None:
        self.assertEqual(short_tool_name("agno:web_search"), "web_search")
        self.assertEqual(short_tool_name("web_search"), "web_search")

    def test_initial_answer_budget_stops_research_but_not_local_actions(self) -> None:
        self.assertTrue(
            should_stop_for_initial_answer(
                has_voice_facts=True,
                user_text="最近世界杯结果是什么",
                elapsed_seconds=12,
                budget_seconds=10,
                tool_name="web_search",
            )
        )
        self.assertFalse(
            should_stop_for_initial_answer(
                has_voice_facts=True,
                user_text="最近世界杯结果是什么",
                elapsed_seconds=12,
                budget_seconds=10,
                tool_name="trigger_fast_followup",
            )
        )
        self.assertFalse(
            should_stop_for_initial_answer(
                has_voice_facts=True,
                user_text="打开浏览器播放视频",
                elapsed_seconds=12,
                budget_seconds=10,
                tool_name="open_url_in_browser",
            )
        )

    def test_tool_missing_requirements_follows_plan_order(self) -> None:
        plan = {
            "steps": [
                {"kind": "tool", "order": 1, "suggested_tools": ["web_search"]},
                {"kind": "tool", "order": 2, "suggested_tools": ["open_url_in_browser"]},
                {"kind": "speak", "order": 3},
            ]
        }
        self.assertEqual(tool_missing_requirements("open_url_in_browser", plan, set()), ["web_search"])
        self.assertEqual(tool_missing_requirements("open_url_in_browser", plan, {"web_search"}), [])
        self.assertEqual(tool_missing_requirements("trigger_fast_followup", plan, set()), ["web_search"])

    def test_should_dedupe_completed_action_only_for_local_actions(self) -> None:
        completed = {"abc"}
        self.assertTrue(should_dedupe_completed_action("arrange_workspace", "abc", completed))
        self.assertFalse(should_dedupe_completed_action("web_search", "abc", completed))
        self.assertFalse(should_dedupe_completed_action("arrange_workspace", "missing", completed))

    def test_classify_followup_action_prioritizes_thresholds_then_content(self) -> None:
        self.assertEqual(classify_followup_action("有结果了", 0, interrupt_threshold=1, speak_threshold=1), "defer")
        self.assertEqual(classify_followup_action("有结果了", 1, interrupt_threshold=1, speak_threshold=3), "context_only")
        self.assertEqual(classify_followup_action("后台任务已完成", 3, interrupt_threshold=1, speak_threshold=3), "suppress_status")
        self.assertEqual(classify_followup_action("Traceback: failed", 3, interrupt_threshold=1, speak_threshold=3), "error_fallback")
        self.assertEqual(classify_followup_action("我查到特朗普最近参加了活动", 3, interrupt_threshold=1, speak_threshold=3), "speak")


class FallbackContextTests(unittest.TestCase):
    def test_recent_tool_context_from_rows_tolerates_bad_json(self) -> None:
        context = recent_tool_context_from_rows(
            [
                {"tool_name": "web_search", "ok": 1, "arguments_json": '{"query":"世界杯"}', "result_json": '{"title":"结果"}'},
                {"tool_name": "fetch_url", "ok": 0, "arguments_json": "{bad", "result_json": ""},
            ]
        )
        self.assertEqual(context[0]["arguments"], {"query": "世界杯"})
        self.assertEqual(context[0]["result"], {"title": "结果"})
        self.assertEqual(context[1]["arguments"], {})
        self.assertEqual(context[1]["result"], {})
        self.assertFalse(context[1]["ok"])

    def test_no_followup_fallback_prompt_includes_user_text_and_tool_facts(self) -> None:
        prompt, facts = no_followup_fallback_prompt(
            "最近世界杯结果是什么",
            [
                {
                    "tool": "web_search",
                    "ok": True,
                    "arguments": {"query": "世界杯 最近结果"},
                    "result": {"results": [{"title": "世界杯结果", "snippet": "阿根廷夺冠"}]},
                }
            ],
        )
        self.assertIn("最近世界杯结果是什么", prompt)
        self.assertIn("阿根廷夺冠", prompt)
        self.assertIn("web_search", facts)

    def test_no_followup_fallback_prompt_requires_direct_yes_no_answer(self) -> None:
        prompt, _ = no_followup_fallback_prompt(
            "圣保罗下雨么",
            [
                {
                    "tool": "daily_action",
                    "ok": True,
                    "arguments": {"action": "weather", "target": "圣保罗"},
                    "result": {"action": "weather", "weather": {"ok": True, "description": "Sunny", "temperature_c": "23"}},
                }
            ],
        )
        self.assertIn("用户问题", prompt)
        self.assertIn("下/不下", prompt)
        self.assertIn("不要自动播完整天气报告", prompt)

    def test_direct_fallback_response_summarizes_daily_weather_without_model(self) -> None:
        response = direct_fallback_response_from_tools(
            "今天天气怎么样",
            [
                {
                    "tool": "daily_action",
                    "ok": True,
                    "arguments": {"action": "weather"},
                    "result": {
                        "action": "weather",
                        "weather": {
                            "ok": True,
                            "resolved_location": "Santo Amaro, Brazil",
                            "temperature_c": "12",
                            "humidity_percent": "94",
                            "wind_kmph": "6",
                            "description": "Sunny",
                        },
                    },
                }
            ],
        )
        self.assertIn("Santo Amaro", response)
        self.assertIn("12度", response)
        self.assertIn("湿度94%", response)

    def test_direct_fallback_response_reports_codex_delegate_started(self) -> None:
        response = direct_fallback_response_from_tools(
            "Debug一下植物大战僵尸",
            [
                {
                    "tool": "computer_action",
                    "ok": True,
                    "arguments": {"action": "delegate_to_codex"},
                    "result": {
                        "ok": True,
                        "action": "delegate_to_codex",
                        "executor": "codex",
                        "_summary": "Codex 开工了",
                    },
                }
            ],
        )
        self.assertEqual(response, "Codex 开工了")

    def test_direct_fallback_response_answers_rain_question_briefly(self) -> None:
        response = direct_fallback_response_from_tools(
            "明天下雨了吗",
            [
                {
                    "tool": "daily_action",
                    "ok": True,
                    "arguments": {"action": "weather"},
                    "result": {
                        "action": "weather",
                        "weather": {
                            "ok": True,
                            "resolved_location": "São Paulo, Brazil",
                            "temperature_c": "23",
                            "description": "Sunny",
                        },
                    },
                }
            ],
        )
        self.assertEqual(response, "不下。")

    def test_direct_fallback_response_prefers_success_over_weather_failure(self) -> None:
        response = direct_fallback_response_from_tools(
            "明天下雨了吗",
            [
                {
                    "tool": "daily_action",
                    "ok": False,
                    "arguments": {"action": "weather", "target": "Avenida Engenheiro Luís Carlos Berrini"},
                    "result": {"action": "weather", "target": "Avenida Engenheiro Luís Carlos Berrini", "weather": {"ok": False, "location": "Avenida Engenheiro Luís Carlos Berrini"}},
                },
                {
                    "tool": "daily_action",
                    "ok": True,
                    "arguments": {"action": "weather", "target": "São Paulo, Brazil"},
                    "result": {"action": "weather", "target": "São Paulo, Brazil", "weather": {"ok": True, "resolved_location": "Santo Amaro, Brazil", "temperature_c": "23", "description": "Sunny"}},
                },
            ],
        )
        self.assertEqual(response, "不下。")
        self.assertNotIn("地点没识别对", response)

    def test_direct_fallback_response_reports_weather_location_failure(self) -> None:
        response = direct_fallback_response_from_tools(
            "今天我孙子试天气怎么样",
            [
                {
                    "tool": "daily_action",
                    "ok": False,
                    "arguments": {"action": "weather", "target": "我孙子试"},
                    "result": {
                        "ok": False,
                        "action": "weather",
                        "target": "我孙子试",
                        "weather": {"ok": False, "location": "我孙子试", "error": "location not found"},
                    },
                }
            ],
        )
        self.assertEqual(response, "地点没识别对，我听成了我孙子试。")

    def test_nested_agno_search_result_exposes_video_urls(self) -> None:
        item = {
            "tool": "agno:web_search",
            "ok": True,
            "arguments": {"query": "青葉世子 MV"},
            "result": {
                "result": (
                    '{"results":[{"title":"MV",'
                    '"url":"https://music.youtube.com/watch?v=zs1xWg4CCYI",'
                    '"snippet":"also https://www.youtube.com/watch?v=YrXlDDvft8Q"}]}'
                )
            },
        }
        urls = extract_urls_from_value(item)
        self.assertIn("https://music.youtube.com/watch?v=zs1xWg4CCYI", urls)
        self.assertTrue(any(looks_like_video_url(url) for url in urls))


class PlanRecoveryTests(unittest.TestCase):
    def test_open_url_uses_verified_url_when_plan_has_no_url(self) -> None:
        args = plan_recovery_tool_args(
            "open_url_in_browser",
            {"kind": "tool", "order": 2, "suggested_tools": ["open_url_in_browser"]},
            "打开刚才找到的网页",
            {"steps": []},
            ["https://example.com/page"],
        )
        self.assertEqual(args, {"url": "https://example.com/page"})

    def test_open_url_is_skipped_without_verified_url(self) -> None:
        args = plan_recovery_tool_args(
            "open_url_in_browser",
            {"kind": "tool", "order": 2, "suggested_tools": ["open_url_in_browser"]},
            "打开刚才找到的网页",
            {"steps": []},
            [],
        )
        self.assertIsNone(args)

    def test_open_url_disables_fullscreen_for_workspace_arrange_requests(self) -> None:
        args = plan_recovery_tool_args(
            "open_url_in_browser",
            {
                "kind": "tool",
                "order": 2,
                "suggested_tools": ["open_url_in_browser"],
                "arguments": {"url": "https://youtube.com/watch?v=abc", "fullscreen": True, "video_fullscreen": True},
            },
            "打开这个视频，然后把 Chrome 和 Codex 并排",
            {"steps": []},
            [],
        )
        self.assertFalse(args["fullscreen"])
        self.assertFalse(args["video_fullscreen"])

    def test_manual_window_layout_osascript_is_skipped(self) -> None:
        args = plan_recovery_tool_args(
            "run_osascript",
            {
                "kind": "tool",
                "order": 1,
                "suggested_tools": ["run_osascript"],
                "arguments": {"script": 'tell application "System Events" to set position of window 1 to {0, 0}'},
            },
            "把窗口排一下",
            {"steps": []},
            [],
        )
        self.assertIsNone(args)


class ToolRuntimeTests(unittest.TestCase):
    def test_callable_tool_map_keeps_named_callables_only(self) -> None:
        def web_search() -> None:
            return None

        tools = callable_tool_map([web_search, "not callable", object()])
        self.assertEqual(list(tools), ["web_search"])
        self.assertIs(tools["web_search"], web_search)

    def test_voice_tools_expose_only_fat_tools_to_agno(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "test-session")
            tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=Path(tmp) / "tools"), store))
        self.assertEqual(
            list(tools),
            ["daily_action", "computer_action", "web_search", "search_news", "trigger_fast_followup"],
        )
        self.assertNotIn("run_python_code", tools)
        self.assertNotIn("front_note", tools)

    def test_daily_action_current_address_returns_structured_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "test-session")
            tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=Path(tmp)), store))
            result = json.loads(tools["daily_action"]("map", "", {"mode": "current_address", "timeout_seconds": 0.2}))
        self.assertEqual(result["action"], "map")
        self.assertIn("ok", result)

    def test_daily_action_rejects_garbage_weather_location_before_wttr(self) -> None:
        original = tool_registry.current_address
        tool_registry.current_address = lambda timeout_seconds=2.0: {"ok": False, "error": "location unavailable"}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "test-session")
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=Path(tmp)), store))
                result = json.loads(tools["daily_action"]("weather", "'s is what", {"timeout_seconds": 0.1}))
        finally:
            tool_registry.current_address = original
        self.assertFalse(result["ok"])
        self.assertIn("地点没识别对", result["error"])

    def test_reminder_due_datetime_parses_chinese_relative_time(self) -> None:
        now = dt.datetime(2026, 6, 21, 10, 0, tzinfo=dt.datetime.now().astimezone().tzinfo)
        tomorrow = tool_registry._reminder_due_datetime("明天", now=now)
        tomorrow_afternoon = tool_registry._reminder_due_datetime("明天下午", now=now)
        tonight = tool_registry._reminder_due_datetime("今晚", now=now)
        fifteen_minutes = tool_registry._reminder_due_datetime("15分钟后", now=now)
        two_hours = tool_registry._reminder_due_datetime("两个小时后", now=now)
        two_hours_after = tool_registry._reminder_due_datetime("两小时之后", now=now)
        half_hour = tool_registry._reminder_due_datetime("半小时后", now=now)
        day_after_tomorrow_eight = tool_registry._reminder_due_datetime("后天八点", now=now)
        self.assertIsNotNone(tomorrow)
        self.assertIsNotNone(tomorrow_afternoon)
        self.assertIsNotNone(tonight)
        self.assertIsNotNone(fifteen_minutes)
        self.assertIsNotNone(two_hours)
        self.assertIsNotNone(two_hours_after)
        self.assertIsNotNone(half_hour)
        self.assertIsNotNone(day_after_tomorrow_eight)
        self.assertEqual((tomorrow.month, tomorrow.day, tomorrow.hour, tomorrow.minute), (6, 22, 9, 0))
        self.assertEqual((tomorrow_afternoon.month, tomorrow_afternoon.day, tomorrow_afternoon.hour, tomorrow_afternoon.minute), (6, 22, 15, 0))
        self.assertEqual((tonight.month, tonight.day, tonight.hour, tonight.minute), (6, 21, 20, 0))
        self.assertEqual((fifteen_minutes.month, fifteen_minutes.day, fifteen_minutes.hour, fifteen_minutes.minute), (6, 21, 10, 15))
        self.assertEqual((two_hours.month, two_hours.day, two_hours.hour, two_hours.minute), (6, 21, 12, 0))
        self.assertEqual((two_hours_after.month, two_hours_after.day, two_hours_after.hour, two_hours_after.minute), (6, 21, 12, 0))
        self.assertEqual((half_hour.month, half_hour.day, half_hour.hour, half_hour.minute), (6, 21, 10, 30))
        self.assertEqual((day_after_tomorrow_eight.month, day_after_tomorrow_eight.day, day_after_tomorrow_eight.hour, day_after_tomorrow_eight.minute), (6, 23, 8, 0))

    def test_coding_action_submit_task_launches_fake_codex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            codex = fake_bin / "codex"
            codex.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$@\" > \"$PWD/codex-argv.txt\"\n"
                "printf '%s\\n' \"$VIRTUAL_ENV\" > \"$PWD/codex-venv.txt\"\n"
                "printf '%s\\n' \"$CODEX_HOME\" > \"$PWD/codex-home.txt\"\n"
                "sleep 2\n",
                encoding="utf-8",
            )
            codex.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            old_kenney = os.environ.get("GJALLARHORN_KENNEY_PLATFORMER_ASSET_DIR")
            os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
            fake_kenney = tmp_path / "kenney_pixel-platformer"
            (fake_kenney / "Tilemap").mkdir(parents=True)
            (fake_kenney / "Tiled").mkdir(parents=True)
            (fake_kenney / "Tilemap" / "tilemap_packed.png").write_bytes(b"fake-png")
            (fake_kenney / "Tiled" / "tileset-tiles.tsx").write_text("<tileset/>", encoding="utf-8")
            os.environ["GJALLARHORN_KENNEY_PLATFORMER_ASSET_DIR"] = str(fake_kenney)
            try:
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                isolated_codex_home = tmp_path / "jen-codex"
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools", coding_codex_home=isolated_codex_home), store))
                self.assertIn("computer_action", tools)
                self.assertNotIn("coding_action", tools)
                self.assertNotIn("python_action", tools)
                result = json.loads(tools["computer_action"](
                    "develop_app",
                    "测试开发",
                    {"prompt": "改一下 README", "cwd": str(tmp_path), "timeout_seconds": 0.1},
                ))
                self.assertTrue(result["ok"])
                self.assertTrue(result["launched"])
                self.assertTrue(result["task_id"])
                self.assertEqual(result["status"], "running")
                self.assertGreater(result["pid"], 0)
                self.assertEqual(result["model"], "gpt-5.3-codex-spark")
                self.assertTrue(Path(result["stdout_log"]).exists())
                self.assertTrue(Path(result["stderr_log"]).exists())
                self.assertTrue(Path(result["prompt_path"]).exists())
                prompt_text = Path(result["prompt_path"]).read_text(encoding="utf-8")
                self.assertIn("uv run --active --no-sync", prompt_text)
                self.assertIn("pywebview", prompt_text)
                self.assertIn("Typer", prompt_text)
                self.assertIn("Agno", prompt_text)
                self.assertIn("InquirerPy", prompt_text)
                self.assertIn("Do not use tkinter, pygame, curses, terminal-only UI", prompt_text)
                self.assertIn("terminal printout", prompt_text)
                self.assertIn("do not use ps, osascript, browser fallback", prompt_text)
                self.assertIn("host voice service will launch and verify", prompt_text)
                self.assertIn("Task workspace:", prompt_text)
                self.assertIn("Active host venv:", prompt_text)
                self.assertIn("VIRTUAL_ENV=", prompt_text)
                self.assertIn("Package caches are opaque", prompt_text)
                self.assertIn("Never run `find`, `rg`, `ls`, `du`, or `git status` against `/Users`", prompt_text)
                self.assertIn("Do not use Phaser 4 unless the user explicitly asks for Phaser 4", prompt_text)
                self.assertIn("do not search cache directories", prompt_text)
                self.assertIn("Staged Kenney platformer assets:", prompt_text)
                self.assertIn(".gjallarhorn/assets/kenney_pixel-platformer", prompt_text)
                self.assertIn("gjallarhorn_asset_manifest.json", prompt_text)
                self.assertNotIn("first try the local npm cache", prompt_text)
                self.assertIn(str(resolve_host_venv()), prompt_text)
                self.assertNotIn("/Users/a1234/.pyenv/versions/3.12.4\n", prompt_text)
                self.assertIn("改一下 README", prompt_text)
                staged_asset = Path(result["cwd"]) / ".gjallarhorn" / "assets" / "kenney_pixel-platformer" / "Tilemap" / "tilemap_packed.png"
                self.assertTrue(staged_asset.exists())
                staged_manifest = staged_asset.parents[1] / "gjallarhorn_asset_manifest.json"
                manifest = json.loads(staged_manifest.read_text(encoding="utf-8"))
                self.assertEqual(manifest["version"], 2)
                self.assertTrue(validate_manifest(manifest)["ok"])
                self.assertIn("pink_player", manifest["semantic_groups"])
                self.assertIn("flying_enemy", manifest["semantic_groups"])
                self.assertIn("blue_pipe", manifest["semantic_groups"])
                self.assertEqual(manifest["semantic_groups"]["ground_grass"]["frames"], [1, 2, 3, 21, 22, 23])
                self.assertIn("sky_background", manifest["semantic_groups"])
                self.assertTrue(manifest["tiles"])
                self.assertTrue(manifest["autotile_rules"])
                self.assertIn("animations", manifest)
                self.assertEqual(manifest["animations"]["flying_enemy_flap"]["frames"], [24, 25, 26])
                self.assertEqual(manifest["animations"]["flying_enemy_flap"]["fps"], 8)
                self.assertEqual(manifest["animations"]["player_pink_walk"]["group"], "pink_player")
                argv_path = tmp_path / "codex-argv.txt"
                for _ in range(20):
                    if argv_path.exists():
                        break
                    time.sleep(0.05)
                argv_lines = argv_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(argv_lines[:3], ["--ask-for-approval", "never", "exec"])
                self.assertIn("--skip-git-repo-check", argv_lines)
                self.assertIn("--sandbox", argv_lines)
                venv_path = tmp_path / "codex-venv.txt"
                for _ in range(20):
                    if venv_path.exists():
                        break
                    time.sleep(0.05)
                self.assertEqual(venv_path.read_text(encoding="utf-8").strip(), str(resolve_host_venv()))
                codex_home_path = tmp_path / "codex-home.txt"
                for _ in range(20):
                    if codex_home_path.exists():
                        break
                    time.sleep(0.05)
                self.assertEqual(codex_home_path.read_text(encoding="utf-8").strip(), str(isolated_codex_home.resolve()))
                self.assertEqual(result["codex_home"], str(isolated_codex_home.resolve()))
                config_text = (isolated_codex_home / "config.toml").read_text(encoding="utf-8")
                self.assertIn("[mcp_servers.context7]", config_text)
                self.assertIn("@upstash/context7-mcp", config_text)
                self.assertTrue((isolated_codex_home / "skills" / "desktop-mini-game" / "SKILL.md").exists())
                self.assertTrue((isolated_codex_home / "skills" / "phaser4-game" / "SKILL.md").exists())
                self.assertTrue((isolated_codex_home / "skills" / "kenney-platformer-assets" / "SKILL.md").exists())
                try:
                    os.kill(int(result["pid"]), 15)
                except Exception:
                    pass
            finally:
                os.environ["PATH"] = old_path
                if old_kenney is None:
                    os.environ.pop("GJALLARHORN_KENNEY_PLATFORMER_ASSET_DIR", None)
                else:
                    os.environ["GJALLARHORN_KENNEY_PLATFORMER_ASSET_DIR"] = old_kenney

    def test_coding_action_explicit_antigravity_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            agy = fake_bin / "agy"
            agy.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$PWD/agy-argv.txt\"\n/bin/sleep 2\n", encoding="utf-8")
            agy.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            old_model = os.environ.pop("GJALLARHORN_ANTIGRAVITY_MODEL", None)
            os.environ["PATH"] = str(fake_bin)
            try:
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools"), store))
                result = json.loads(tools["computer_action"](
                    "develop_app",
                    "测试开发",
                    {"prompt": "改一下 README", "cwd": str(tmp_path), "timeout_seconds": 0.1, "executor": "antigravity"},
                ))
                self.assertFalse(result["ok"])
                self.assertEqual(result["executor"], "antigravity")
                self.assertEqual(result["executor_selection_reason"], "explicit")
                self.assertFalse(result["executor_fallback"])
                self.assertIn("disabled", result["error"])
                self.assertFalse((tmp_path / "agy-argv.txt").exists())
            finally:
                os.environ["PATH"] = old_path
                if old_model is not None:
                    os.environ["GJALLARHORN_ANTIGRAVITY_MODEL"] = old_model

    def test_coding_action_defaults_to_codex_for_repeated_submits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            for name in ("codex", "agy"):
                script = fake_bin / name
                script.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$PWD/{name}-argv.txt\"\n/bin/sleep 2\n", encoding="utf-8")
                script.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = str(fake_bin)
            try:
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools"), store))
                first = json.loads(tools["computer_action"]("develop_app", "第一个开发", {"prompt": "改 README 1", "cwd": str(tmp_path), "timeout_seconds": 0.1}))
                second = json.loads(tools["computer_action"]("develop_app", "第二个开发", {"prompt": "改 README 2", "cwd": str(tmp_path), "timeout_seconds": 0.1}))
                third = json.loads(tools["computer_action"]("develop_app", "第三个开发", {"prompt": "改 README 3", "cwd": str(tmp_path), "timeout_seconds": 0.1}))
                self.assertEqual([first["executor"], second["executor"], third["executor"]], ["codex", "codex", "codex"])
                self.assertEqual([first["executor_selection_reason"], second["executor_selection_reason"], third["executor_selection_reason"]], ["default", "default", "default"])
                self.assertFalse((tmp_path / "agy-argv.txt").exists())
                for result in [first, second, third]:
                    try:
                        os.kill(int(result["pid"]), 15)
                    except Exception:
                        pass
            finally:
                os.environ["PATH"] = old_path

    def test_coding_action_default_codex_ignores_previous_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            codex = fake_bin / "codex"
            codex.write_text("#!/bin/sh\n/bin/sleep 2\n", encoding="utf-8")
            codex.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = str(fake_bin)
            try:
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                store.create_coding_task({"run_id": "previous", "pid": 0, "status": "running", "executor": "codex"})
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools"), store))
                result = json.loads(tools["computer_action"]("develop_app", "测试开发", {"prompt": "改 README", "cwd": str(tmp_path), "timeout_seconds": 0.1}))
                self.assertTrue(result["ok"])
                self.assertEqual(result["executor"], "codex")
                self.assertEqual(result["executor_selection_reason"], "default")
                self.assertFalse(result["executor_fallback"])
                try:
                    os.kill(int(result["pid"]), 15)
                except Exception:
                    pass
            finally:
                os.environ["PATH"] = old_path

    def test_coding_action_explicit_antigravity_missing_does_not_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            codex = fake_bin / "codex"
            codex.write_text("#!/bin/sh\n/bin/sleep 2\n", encoding="utf-8")
            codex.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = str(fake_bin)
            try:
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools"), store))
                result = json.loads(tools["computer_action"]("develop_app", "测试开发", {"prompt": "改 README", "cwd": str(tmp_path), "timeout_seconds": 0.1, "executor": "antigravity"}))
                self.assertFalse(result["ok"])
                self.assertFalse(result["executor_fallback"])
                self.assertEqual(result["executor"], "antigravity")
                self.assertIn("disabled", result["error"])
            finally:
                os.environ["PATH"] = old_path

    def test_computer_action_delegate_to_codex_forces_codex_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            codex = fake_bin / "codex"
            codex.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$PWD/codex-argv.txt\"\n/bin/sleep 2\n", encoding="utf-8")
            codex.chmod(0o755)
            agy = fake_bin / "agy"
            agy.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$PWD/agy-argv.txt\"\n/bin/sleep 2\n", encoding="utf-8")
            agy.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = str(fake_bin)
            try:
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools"), store))
                result = json.loads(tools["computer_action"](
                    "delegate_to_codex",
                    "未知任务",
                    {"prompt": "完全不知道怎么做的复杂任务", "timeout_seconds": 0.1},
                ))
                self.assertTrue(result["ok"])
                self.assertEqual(result["action"], "delegate_to_codex")
                self.assertEqual(result["executor"], "codex")
                self.assertEqual(result["executor_selection_reason"], "explicit")
                for _ in range(20):
                    if (Path(result["cwd"]) / "codex-argv.txt").exists():
                        break
                    time.sleep(0.05)
                self.assertTrue((Path(result["cwd"]) / "codex-argv.txt").exists())
                self.assertFalse((Path(result["cwd"]) / "agy-argv.txt").exists())
                try:
                    os.kill(int(result["pid"]), 15)
                except Exception:
                    pass
            finally:
                os.environ["PATH"] = old_path

    def test_coding_action_submit_task_defaults_to_managed_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            codex = fake_bin / "codex"
            codex.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$PWD/codex-argv.txt\"\nprintf '%s\\n' \"$VIRTUAL_ENV\" > \"$PWD/codex-venv.txt\"\nsleep 2\n", encoding="utf-8")
            codex.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
            try:
                tool_workdir = tmp_path / "tools"
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tool_workdir), store))
                result = json.loads(tools["computer_action"]("develop_app", "写个小游戏", {"timeout_seconds": 0.1}))
                self.assertTrue(result["ok"])
                workspace = Path(result["workspace"])
                self.assertTrue(workspace.exists())
                self.assertEqual(workspace.parent, (tool_workdir / "coding_workspaces").resolve())
                self.assertEqual(result["cwd"], str(workspace))
                pyproject_text = (workspace / "pyproject.toml").read_text(encoding="utf-8")
                self.assertIn("dependencies = [", pyproject_text)
                self.assertNotIn('"pywebview"', pyproject_text)
                self.assertNotIn('"InquirerPy"', pyproject_text)
                self.assertIn("package = false", pyproject_text)
                prompt_text = Path(result["prompt_path"]).read_text(encoding="utf-8")
                self.assertIn(str(workspace), prompt_text)
                self.assertIn("Workspace context and callable local services:", prompt_text)
                self.assertIn("minimal pyproject.toml", prompt_text)
                self.assertIn("Do not use bare python", prompt_text)
                self.assertIn("do not run uv sync", prompt_text)
                self.assertIn("VIRTUAL_ENV=", prompt_text)
                self.assertIn(str(resolve_host_venv()), prompt_text)
                argv_path = workspace / "codex-argv.txt"
                for _ in range(20):
                    if argv_path.exists():
                        break
                    time.sleep(0.05)
                argv_lines = argv_path.read_text(encoding="utf-8").splitlines()
                cd_index = argv_lines.index("--cd")
                self.assertEqual(argv_lines[cd_index + 1], str(workspace))
                try:
                    os.kill(int(result["pid"]), 15)
                except Exception:
                    pass
            finally:
                os.environ["PATH"] = old_path

    def test_coding_action_reuses_similar_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            codex = fake_bin / "codex"
            codex.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$PWD/codex-argv.txt\"\nsleep 2\n", encoding="utf-8")
            codex.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
            try:
                tool_workdir = tmp_path / "tools"
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tool_workdir), store))
                first = json.loads(tools["computer_action"]("develop_app", "写个吃豆人游戏", {"timeout_seconds": 0.1}))
                second = json.loads(tools["computer_action"]("develop_app", "把吃豆人加暂停按钮", {"timeout_seconds": 0.1}))
                self.assertTrue(first["ok"])
                self.assertTrue(second["ok"])
                self.assertFalse(first["workspace_reused"])
                self.assertTrue(second["workspace_reused"])
                self.assertEqual(first["workspace_id"], second["workspace_id"])
                self.assertEqual(first["workspace"], second["workspace"])
                self.assertGreaterEqual(second["workspace_score"], 0.35)
                self.assertTrue(Path(first["workspace_manifest_path"]).exists())
                manifest = read_manifest(Path(first["workspace"]))
                self.assertIsNotNone(manifest)
                self.assertEqual(manifest["workspace_id"], first["workspace_id"])
                for result in [first, second]:
                    try:
                        os.kill(int(result["pid"]), 15)
                    except Exception:
                        pass
            finally:
                os.environ["PATH"] = old_path

    def test_coding_action_debug_reuses_similar_workspace_with_asr_typos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            codex = fake_bin / "codex"
            codex.write_text("#!/bin/sh\nsleep 2\n", encoding="utf-8")
            codex.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
            try:
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools"), store))
                first = json.loads(tools["computer_action"]("develop_app", "开发一个植物大战僵尸", {"timeout_seconds": 0.1}))
                second = json.loads(tools["computer_action"]("develop_app", "Dbug 植物大转僵尸", {"timeout_seconds": 0.1}))
                self.assertTrue(first["ok"])
                self.assertTrue(second["ok"])
                self.assertFalse(first["workspace_reused"])
                self.assertTrue(second["workspace_reused"])
                self.assertEqual(first["workspace_id"], second["workspace_id"])
                self.assertGreaterEqual(second["workspace_score"], 0.35)
                for result in [first, second]:
                    try:
                        os.kill(int(result["pid"]), 15)
                    except Exception:
                        pass
            finally:
                os.environ["PATH"] = old_path

    def test_coding_action_default_reuses_similar_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            codex = fake_bin / "codex"
            codex.write_text("#!/bin/sh\nsleep 2\n", encoding="utf-8")
            codex.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
            try:
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools"), store))
                first = json.loads(tools["computer_action"]("develop_app", "写个猪头在屏幕前乱碰", {"timeout_seconds": 0.1}))
                second = json.loads(tools["computer_action"]("develop_app", "写个猪头在屏幕前乱跳", {"timeout_seconds": 0.1}))
                self.assertTrue(first["ok"])
                self.assertTrue(second["ok"])
                self.assertFalse(first["workspace_reused"])
                self.assertTrue(second["workspace_reused"])
                self.assertEqual(first["workspace_id"], second["workspace_id"])
                self.assertEqual(first["workspace"], second["workspace"])
                for result in [first, second]:
                    try:
                        os.kill(int(result["pid"]), 15)
                    except Exception:
                        pass
            finally:
                os.environ["PATH"] = old_path

    def test_coding_action_task_mode_controls_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            codex = fake_bin / "codex"
            codex.write_text("#!/bin/sh\nsleep 2\n", encoding="utf-8")
            codex.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
            try:
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools"), store))
                first = json.loads(tools["computer_action"]("develop_app", "写个地图 app", {"timeout_seconds": 0.1}))
                forced_new = json.loads(tools["computer_action"]("develop_app", "地图联动窗口", {"task_mode": "create_new", "prompt": "新做一个跟地图 app 交互的窗口", "timeout_seconds": 0.1}))
                forced_reuse = json.loads(tools["computer_action"]("develop_app", "地图 app", {"task_mode": "extend_existing", "prompt": "给地图 app 加一个按钮", "timeout_seconds": 0.1}))
                self.assertTrue(first["ok"])
                self.assertTrue(forced_new["ok"])
                self.assertTrue(forced_reuse["ok"])
                self.assertNotEqual(first["workspace_id"], forced_new["workspace_id"])
                self.assertEqual(first["workspace_id"], forced_reuse["workspace_id"])
                self.assertTrue(forced_reuse["workspace_reused"])
                for result in [first, forced_new, forced_reuse]:
                    try:
                        os.kill(int(result["pid"]), 15)
                    except Exception:
                        pass
            finally:
                os.environ["PATH"] = old_path

    def test_coding_action_explicit_new_workspace_overrides_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            codex = fake_bin / "codex"
            codex.write_text("#!/bin/sh\nsleep 2\n", encoding="utf-8")
            codex.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
            try:
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools"), store))
                first = json.loads(tools["computer_action"]("develop_app", "写个吃豆人游戏", {"timeout_seconds": 0.1}))
                second = json.loads(tools["computer_action"]("develop_app", "新建一个吃豆人游戏", {"timeout_seconds": 0.1}))
                self.assertTrue(first["ok"])
                self.assertTrue(second["ok"])
                self.assertNotEqual(first["workspace_id"], second["workspace_id"])
                self.assertFalse(second["workspace_reused"])
                for result in [first, second]:
                    try:
                        os.kill(int(result["pid"]), 15)
                    except Exception:
                        pass
            finally:
                os.environ["PATH"] = old_path

    def test_coding_workspace_dedupe_marks_related_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            root = tmp_path / "tools" / "coding_workspaces"
            left = root / "pvz-a"
            right = root / "pvz-b"
            left.mkdir(parents=True)
            right.mkdir(parents=True)
            index = CodingWorkspaceIndex(store, root)
            left_id = store.upsert_coding_workspace({
                "path": str(left),
                "title": "植物大战僵尸",
                "aliases": ["植物大战僵尸"],
                "tags": ["植物", "僵尸"],
                "summary": "pywebview 塔防游戏",
                "capabilities": ["desktop_app", "game"],
                "entrypoints": [],
                "status": "active",
                "last_task_at": 10,
            })
            right_id = store.upsert_coding_workspace({
                "path": str(right),
                "title": "植物大转僵尸",
                "aliases": ["植物大转僵尸"],
                "tags": ["植物", "僵尸"],
                "summary": "同类塔防游戏",
                "capabilities": ["desktop_app", "game"],
                "entrypoints": [],
                "status": "active",
                "last_task_at": 20,
            })
            updates = index.dedupe_related(threshold=0.35)
            self.assertTrue(updates)
            left_record = index.get(workspace_id=left_id)
            right_record = index.get(workspace_id=right_id)
            self.assertTrue(left.exists())
            self.assertTrue(right.exists())
            related = set(left_record.get("related_workspace_ids") or []) | set(right_record.get("related_workspace_ids") or [])
            self.assertIn(left_id, related)
            self.assertIn(right_id, related)

    def test_coding_action_list_and_inspect_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            workspace_id = store.upsert_coding_workspace({
                "path": str(workspace),
                "title": "吃豆人游戏",
                "aliases": ["Pac-Man"],
                "tags": ["吃豆人", "游戏"],
                "summary": "本地 pywebview 吃豆人小游戏",
                "capabilities": ["desktop_app", "game"],
                "entrypoints": [{"type": "python", "path": "app.py", "role": "app"}],
                "services": [{"name": "game", "url": "http://127.0.0.1:65530", "health_path": "/health"}],
            })
            tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools"), store))
            listed = json.loads(tools["computer_action"]("list_workspaces", "吃豆人", {}))
            inspected = json.loads(tools["computer_action"]("inspect_workspace", "", {"workspace_id": workspace_id}))
        self.assertTrue(listed["ok"])
        self.assertEqual(listed["workspaces"][0]["workspace_id"], workspace_id)
        self.assertGreater(listed["workspaces"][0]["score"], 0)
        self.assertTrue(inspected["ok"])
        self.assertEqual(inspected["workspace"]["workspace_id"], workspace_id)
        self.assertEqual(inspected["workspace"]["services"][0]["status"], "down")

    def test_coding_action_submit_task_uses_target_as_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            codex = fake_bin / "codex"
            codex.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$PWD/codex-argv.txt\"\nsleep 2\n", encoding="utf-8")
            codex.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
            try:
                store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
                tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools"), store))
                result = json.loads(tools["computer_action"]("develop_app", "改一下 README", {"cwd": str(tmp_path), "timeout_seconds": 0.1}))
                self.assertTrue(result["ok"])
                try:
                    os.kill(int(result["pid"]), 15)
                except Exception:
                    pass
            finally:
                os.environ["PATH"] = old_path

    def test_coding_action_submit_task_requires_prompt_or_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "test-session")
            tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=Path(tmp) / "tools"), store))
            result = json.loads(tools["computer_action"]("develop_app", "", {}))
        self.assertFalse(result["ok"])
        self.assertFalse(result["launched"])
        self.assertIn("prompt", result["error"])

    def test_coding_action_submit_task_rejects_bad_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceSessionStore(Path(tmp) / "voice.sqlite", "test-session")
            tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=Path(tmp) / "tools"), store))
            result = json.loads(tools["computer_action"](
                "submit_task",
                "测试开发",
                {"prompt": "改一下 README", "cwd": str(Path(tmp) / "missing")},
            ))
        self.assertFalse(result["ok"])
        self.assertFalse(result["launched"])
        self.assertIn("cwd", result["error"])

    def test_resolve_host_venv_prefers_env_then_repo_venv(self) -> None:
        old_venv = os.environ.get("VIRTUAL_ENV")
        with tempfile.TemporaryDirectory() as tmp:
            fake_venv = Path(tmp) / "env-venv"
            (fake_venv / "bin").mkdir(parents=True)
            (fake_venv / "bin" / "python").write_text("", encoding="utf-8")
            os.environ["VIRTUAL_ENV"] = str(fake_venv)
            try:
                self.assertEqual(resolve_host_venv(), fake_venv.resolve())
                os.environ.pop("VIRTUAL_ENV", None)
                self.assertEqual(resolve_host_venv(), (ROOT / ".venv").resolve())
            finally:
                if old_venv is None:
                    os.environ.pop("VIRTUAL_ENV", None)
                else:
                    os.environ["VIRTUAL_ENV"] = old_venv

    def test_coding_monitor_records_progress_completion_and_speech(self) -> None:
        class FakeSpeech:
            def __init__(self) -> None:
                self.spoken: list[str] = []

            def speak(self, text: str, **_kwargs) -> None:
                self.spoken.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stdout_log = tmp_path / "stdout.jsonl"
            last_message = tmp_path / "last_message.txt"
            stdout_log.write_text(
                "\n".join([
                    json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "我正在检查项目结构。"}}, ensure_ascii=False),
                    json.dumps({"type": "item.completed", "item": {"type": "file_change", "changes": [{"path": str(tmp_path / "app.py"), "kind": "add"}]}}, ensure_ascii=False),
                    json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1}}, ensure_ascii=False),
                ]) + "\n",
                encoding="utf-8",
            )
            last_message.write_text("已经新增 app.py。", encoding="utf-8")
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            task_id = store.create_coding_task({
                "run_id": "run-1",
                "pid": 0,
                "status": "running",
                "target": "测试开发",
                "stdout_log": str(stdout_log),
                "stderr_log": str(tmp_path / "stderr.log"),
                "last_message_path": str(last_message),
                "prompt_path": str(tmp_path / "prompt.txt"),
                "next_speech_at": 0,
            })
            speech = FakeSpeech()
            monitor = CodingTaskMonitor(store, speech, speech_interval_seconds=60)
            monitor.poll_once()
            tasks = store.coding_tasks(statuses=("completed",), limit=10)
            pipeline = store.recent_pipeline(limit=50)
        self.assertEqual(tasks[0]["task_id"], task_id)
        self.assertGreater(tasks[0]["last_offset"], 0)
        self.assertIn("Codex 完成", tasks[0]["last_summary"])
        self.assertTrue(any("Codex 完成" in text for text in speech.spoken))
        self.assertTrue(any(item["kind"] == "coding_task_status" for item in pipeline["items"]))
        self.assertFalse(any(item["content"] == "Codex 完成 · 任务处理完了" for item in pipeline["items"]))
        self.assertTrue(any(item["content"] == "Codex 产物完成" for item in pipeline["items"]))

    def test_coding_monitor_marks_failed_completion_message_as_failed(self) -> None:
        class FakeSpeech:
            def __init__(self) -> None:
                self.spoken: list[str] = []

            def speak(self, text: str, **_kwargs) -> None:
                self.spoken.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stdout_log = tmp_path / "stdout.jsonl"
            last_message = tmp_path / "last_message.txt"
            stdout_log.write_text(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1}}, ensure_ascii=False) + "\n", encoding="utf-8")
            last_message.write_text("当前环境无法成功运行 GUI：ModuleNotFoundError No module named webview。", encoding="utf-8")
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            task_id = store.create_coding_task({
                "run_id": "run-failed",
                "pid": 0,
                "status": "running",
                "target": "测试开发",
                "stdout_log": str(stdout_log),
                "last_message_path": str(last_message),
                "next_speech_at": 0,
            })
            speech = FakeSpeech()
            monitor = CodingTaskMonitor(store, speech, speech_interval_seconds=60)
            monitor.poll_once()
            tasks = store.coding_tasks(statuses=("failed",), limit=10)
        self.assertEqual(tasks[0]["task_id"], task_id)
        self.assertIn("Codex 失败", tasks[0]["last_summary"])
        self.assertTrue(any("Codex 失败" in text for text in speech.spoken))

    def test_coding_monitor_marks_nsscreen_completion_message_as_failed(self) -> None:
        class FakeSpeech:
            def __init__(self) -> None:
                self.spoken: list[str] = []

            def speak(self, text: str, **_kwargs) -> None:
                self.spoken.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stdout_log = tmp_path / "stdout.jsonl"
            last_message = tmp_path / "last_message.txt"
            stdout_log.write_text(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1}}, ensure_ascii=False) + "\n", encoding="utf-8")
            last_message.write_text("已写好，但当前会话是无图形界面环境，AppKit.NSScreen.mainScreen() is None，无法创建 Cocoa 窗口。", encoding="utf-8")
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            task_id = store.create_coding_task({
                "run_id": "run-nsscreen",
                "pid": 0,
                "status": "running",
                "target": "测试开发",
                "stdout_log": str(stdout_log),
                "last_message_path": str(last_message),
                "next_speech_at": 0,
            })
            speech = FakeSpeech()
            monitor = CodingTaskMonitor(store, speech, speech_interval_seconds=60)
            monitor.poll_once()
            tasks = store.coding_tasks(statuses=("failed",), limit=10)
        self.assertEqual(tasks[0]["task_id"], task_id)
        self.assertIn("Codex 失败", tasks[0]["last_summary"])

    def test_coding_monitor_launches_pywebview_workspace_from_host_venv(self) -> None:
        class FakeSpeech:
            def __init__(self) -> None:
                self.spoken: list[str] = []

            def speak(self, text: str, **_kwargs) -> None:
                self.spoken.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_venv = tmp_path / "venv"
            fake_bin = fake_venv / "bin"
            fake_bin.mkdir(parents=True)
            uv = fake_bin / "uv"
            uv.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$VIRTUAL_ENV\" > \"$PWD/used-venv.txt\"\n"
                "printf '%s\\n' \"$@\" > \"$PWD/used-argv.txt\"\n"
                "sleep 5\n",
                encoding="utf-8",
            )
            uv.chmod(0o755)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            (workspace / "app.py").write_text("import webview\nwebview.create_window('x', html='<p>x</p>')\nwebview.start()\n", encoding="utf-8")
            stdout_log = tmp_path / "stdout.jsonl"
            last_message = tmp_path / "last_message.txt"
            stdout_log.write_text(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1}}, ensure_ascii=False) + "\n", encoding="utf-8")
            last_message.write_text("已经新增 app.py。", encoding="utf-8")
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            task_id = store.create_coding_task({
                "run_id": "run-host-launch",
                "pid": 0,
                "status": "running",
                "target": "测试桌面应用",
                "workspace": str(workspace),
                "cwd": str(workspace),
                "stdout_log": str(stdout_log),
                "last_message_path": str(last_message),
                "active_venv": str(fake_venv),
                "next_speech_at": 0,
            })
            speech = FakeSpeech()
            monitor = CodingTaskMonitor(store, speech, speech_interval_seconds=60)
            monitor.runner = CodingAppRunner(venv=fake_venv)
            monitor.poll_once()
            tasks = store.coding_tasks(statuses=("completed",), limit=10)
            self.assertEqual(tasks[0]["task_id"], task_id)
            self.assertIn("已写完并启动 run.sh", tasks[0]["last_summary"])
            self.assertEqual((workspace / "used-venv.txt").read_text(encoding="utf-8").strip(), str(fake_venv))
            self.assertIn("run\n--active\n--no-sync\npython\napp.py", (workspace / "used-argv.txt").read_text(encoding="utf-8"))
            self.assertTrue((workspace / "run.sh").exists())
            self.assertIn("cmd=bash run.sh", (workspace / ".voice_app_run.log").read_text(encoding="utf-8"))
            manifest = read_manifest(workspace)
            self.assertIsNotNone(manifest)
            program = manifest.get("program") or {}
            self.assertEqual(program["kind"], "coding_app")
            self.assertEqual(program["open_method"]["type"], "script")
            self.assertEqual(program["open_method"]["entrypoint"], "run.sh")
            self.assertEqual(program["open_method"]["argv"], ["bash", "run.sh"])
            self.assertEqual(program["open_method"]["env"]["VIRTUAL_ENV"], str(fake_venv))

    def test_coding_runner_treats_gui_window_timeout_as_soft_without_manifest_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            (workspace / "app.py").write_text(
                "import webview\nwebview.create_window('Pac-Man', html='<p>x</p>')\nwebview.start()\n",
                encoding="utf-8",
            )
            runner = CodingAppRunner(launch_wait_seconds=0.1)
            with patch.object(
                coding_monitor,
                "_visible_window_snapshot",
                return_value={"ok": False, "error": "window enumeration timed out for python3"},
            ):
                result = runner._verify_gui_window(workspace)
        self.assertTrue(result["required"])
        self.assertTrue(result["ok"])
        self.assertTrue(result["unverified"])
        self.assertIn("window enumeration timed out", result["error"])

    def test_static_asset_validation_catches_missing_js_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "index.html").write_text(
                "<canvas id='game'></canvas><script src='js/game.js'></script>",
                encoding="utf-8",
            )
            (workspace / "js").mkdir()
            (workspace / "js" / "game.js").write_text(
                "const img = new Image(); img.src = 'assets/kenney/tilemap.png';",
                encoding="utf-8",
            )
            result = validate_workspace_static_assets(workspace)
        self.assertFalse(result["ok"])
        self.assertIn("js/game.js -> assets/kenney/tilemap.png", result["missing"])

    def test_static_asset_validation_accepts_workspace_root_public_asset_from_js(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "index.html").write_text(
                "<script src='js/game.js'></script>",
                encoding="utf-8",
            )
            (workspace / "js").mkdir()
            (workspace / "js" / "game.js").write_text(
                "const img = new Image(); img.src = 'public/assets/kenney/tilemap.png';",
                encoding="utf-8",
            )
            asset = workspace / "public" / "assets" / "kenney" / "tilemap.png"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"png")
            result = validate_workspace_static_assets(workspace)
        self.assertTrue(result["ok"], result)

    def test_computer_action_open_program_uses_registered_open_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_venv = tmp_path / "venv"
            fake_bin = fake_venv / "bin"
            fake_bin.mkdir(parents=True)
            uv = fake_bin / "uv"
            uv.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$VIRTUAL_ENV\" > \"$PWD/open-venv.txt\"\n"
                "printf '%s\\n' \"$@\" > \"$PWD/open-argv.txt\"\n"
                "sleep 5\n",
                encoding="utf-8",
            )
            uv.chmod(0o755)
            workspace = tmp_path / "tools" / "coding_workspaces" / "pig-app"
            workspace.mkdir(parents=True)
            (workspace / "app.py").write_text("import webview\nwebview.create_window('猪头', html='<p>x</p>')\nwebview.start()\n", encoding="utf-8")
            (workspace / "run.sh").write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$VIRTUAL_ENV\" > \"$PWD/open-venv.txt\"\n"
                "printf '%s\\n' \"$@\" > \"$PWD/open-argv.txt\"\n"
                "sleep 5\n",
                encoding="utf-8",
            )
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            workspace_id = store.upsert_coding_workspace({
                "path": str(workspace),
                "title": "弹跳猪头",
                "aliases": ["猪头", "pig"],
                "tags": ["猪头"],
                "summary": "pywebview 猪头动画",
                "capabilities": ["desktop_app"],
                "entrypoints": [{"type": "python", "path": "app.py", "role": "app"}],
                "program": {
                    "program_id": "coding:test-pig",
                    "workspace_id": "test-pig",
                    "name": "弹跳猪头",
                    "aliases": ["弹跳猪头", "猪头", "pig"],
                    "kind": "coding_app",
                    "open_method": {
                        "type": "script",
                        "cwd": str(workspace),
                        "entrypoint": "run.sh",
                        "argv": ["bash", "run.sh"],
                        "env": {"VIRTUAL_ENV": str(fake_venv)},
                    },
                    "window_match": {"app_name": "Python", "title_keywords": ["猪头"]},
                    "capabilities": ["desktop_app"],
                    "status": "ready",
                },
            })
            tools = callable_tool_map(build_voice_tools(SimpleNamespace(tool_workdir=tmp_path / "tools"), store))
            result = json.loads(tools["computer_action"]("open_program", "猪头", {}))
            self.assertTrue(result["ok"])
            self.assertEqual(result["workspace"]["workspace_id"], workspace_id)
            self.assertEqual((workspace / "open-venv.txt").read_text(encoding="utf-8").strip(), str(fake_venv))
            self.assertIn("cmd=bash run.sh", (workspace / ".voice_app_run.log").read_text(encoding="utf-8"))
            try:
                os.kill(int(result["launch"]["pid"]), 15)
            except Exception:
                pass

    def test_coding_monitor_marks_app_launch_failure_failed_with_log(self) -> None:
        class FakeSpeech:
            def __init__(self) -> None:
                self.spoken: list[str] = []

            def speak(self, text: str, **_kwargs) -> None:
                self.spoken.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_venv = tmp_path / "venv"
            fake_bin = fake_venv / "bin"
            fake_bin.mkdir(parents=True)
            uv = fake_bin / "uv"
            uv.write_text("#!/bin/sh\necho 'No module named webview' >&2\nexit 1\n", encoding="utf-8")
            uv.chmod(0o755)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            (workspace / "app.py").write_text("import webview\nwebview.start()\n", encoding="utf-8")
            stdout_log = tmp_path / "stdout.jsonl"
            stdout_log.write_text(json.dumps({"type": "turn.completed"}, ensure_ascii=False) + "\n", encoding="utf-8")
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            task_id = store.create_coding_task({
                "run_id": "run-launch-fail",
                "pid": 0,
                "status": "running",
                "target": "测试桌面应用",
                "workspace": str(workspace),
                "cwd": str(workspace),
                "stdout_log": str(stdout_log),
                "active_venv": str(fake_venv),
                "next_speech_at": 0,
            })
            speech = FakeSpeech()
            monitor = CodingTaskMonitor(store, speech, speech_interval_seconds=60)
            monitor.runner = CodingAppRunner(venv=fake_venv, launch_wait_seconds=1.0)
            monitor.poll_once()
            tasks = store.coding_tasks(statuses=("failed",), limit=10)
            pipeline = store.recent_pipeline(limit=50)
            run_log_exists = (workspace / ".voice_app_run.log").exists()
        self.assertEqual(tasks[0]["task_id"], task_id)
        self.assertIn("写完了但没启动", tasks[0]["last_summary"])
        self.assertTrue(run_log_exists)
        self.assertTrue(any("写完了但没启动" in text for text in speech.spoken))
        self.assertFalse(any(item["content"] == "Codex 完成 · 任务处理完了" for item in pipeline["items"]))

    def test_coding_monitor_periodic_running_speech_is_rate_limited(self) -> None:
        class FakeSpeech:
            def __init__(self) -> None:
                self.spoken: list[str] = []

            def speak(self, text: str, **_kwargs) -> None:
                self.spoken.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stdout_log = tmp_path / "stdout.jsonl"
            stdout_log.write_text(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "我还在处理。"}}, ensure_ascii=False) + "\n", encoding="utf-8")
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            task_id = store.create_coding_task({
                "run_id": "run-2",
                "pid": 0,
                "status": "running",
                "target": "测试开发",
                "stdout_log": str(stdout_log),
                "next_speech_at": 0,
            })
            speech = FakeSpeech()
            monitor = CodingTaskMonitor(store, speech, speech_interval_seconds=60)
            monitor.poll_once()
            monitor.poll_once()
            tasks = store.coding_tasks(statuses=("running",), limit=10)
        self.assertEqual(tasks[0]["task_id"], task_id)
        self.assertEqual(len(speech.spoken), 1)

    def test_coding_monitor_treats_antigravity_plain_output_as_completion(self) -> None:
        class FakeSpeech:
            def __init__(self) -> None:
                self.spoken: list[str] = []

            def speak(self, text: str, **_kwargs) -> None:
                self.spoken.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stdout_log = tmp_path / "stdout.log"
            stdout_log.write_text("已经完成 README 修改。\n", encoding="utf-8")
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            task_id = store.create_coding_task({
                "run_id": "agy-run",
                "pid": 99999999,
                "status": "running",
                "executor": "antigravity",
                "target": "测试开发",
                "stdout_log": str(stdout_log),
                "next_speech_at": 0,
            })
            speech = FakeSpeech()
            monitor = CodingTaskMonitor(store, speech, speech_interval_seconds=60)
            monitor.poll_once()
            tasks = store.coding_tasks(statuses=("completed",), limit=10)
        self.assertEqual(tasks[0]["task_id"], task_id)
        self.assertIn("Antigravity 完成", tasks[0]["last_summary"])
        self.assertTrue(any("Antigravity 完成" in text for text in speech.spoken))

    def test_coding_monitor_treats_antigravity_timeout_output_as_failure(self) -> None:
        class FakeSpeech:
            def __init__(self) -> None:
                self.spoken: list[str] = []

            def speak(self, text: str, **_kwargs) -> None:
                self.spoken.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stdout_log = tmp_path / "stdout.log"
            stdout_log.write_text("Error: timed out waiting for response\n", encoding="utf-8")
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            task_id = store.create_coding_task({
                "run_id": "agy-timeout",
                "pid": 99999999,
                "status": "running",
                "executor": "antigravity",
                "target": "测试开发",
                "stdout_log": str(stdout_log),
                "next_speech_at": 0,
            })
            speech = FakeSpeech()
            monitor = CodingTaskMonitor(store, speech, speech_interval_seconds=60)
            monitor.poll_once()
            tasks = store.coding_tasks(statuses=("failed",), limit=10)
        self.assertEqual(tasks[0]["task_id"], task_id)
        self.assertIn("Antigravity 失败", tasks[0]["last_summary"])
        self.assertIn("超时", tasks[0]["last_summary"])

    def test_coding_monitor_treats_antigravity_print_timeout_misroute_as_failure(self) -> None:
        class FakeSpeech:
            def __init__(self) -> None:
                self.spoken: list[str] = []

            def speak(self, text: str, **_kwargs) -> None:
                self.spoken.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stdout_log = tmp_path / "stdout.log"
            stdout_log.write_text("The --print-timeout flag is a command-line option for the Antigravity CLI (`agy`).\n", encoding="utf-8")
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            task_id = store.create_coding_task({
                "run_id": "agy-misroute",
                "pid": 99999999,
                "status": "running",
                "executor": "antigravity",
                "target": "测试开发",
                "stdout_log": str(stdout_log),
                "next_speech_at": 0,
            })
            speech = FakeSpeech()
            monitor = CodingTaskMonitor(store, speech, speech_interval_seconds=60)
            monitor.poll_once()
            tasks = store.coding_tasks(statuses=("failed",), limit=10)
        self.assertEqual(tasks[0]["task_id"], task_id)
        self.assertIn("偏离开发任务", tasks[0]["last_summary"])

    def test_coding_monitor_reads_antigravity_executor_log_failures(self) -> None:
        class FakeSpeech:
            def __init__(self) -> None:
                self.spoken: list[str] = []

            def speak(self, text: str, **_kwargs) -> None:
                self.spoken.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stdout_log = tmp_path / "stdout.log"
            stdout_log.write_text("", encoding="utf-8")
            executor_log = tmp_path / "antigravity.log"
            executor_log.write_text("RESOURCE_EXHAUSTED (code 429): Individual quota reached.\n", encoding="utf-8")
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            task_id = store.create_coding_task({
                "run_id": "agy-quota",
                "pid": 99999999,
                "status": "running",
                "executor": "antigravity",
                "target": "测试开发",
                "stdout_log": str(stdout_log),
                "executor_log": str(executor_log),
                "next_speech_at": 0,
            })
            speech = FakeSpeech()
            monitor = CodingTaskMonitor(store, speech, speech_interval_seconds=60)
            monitor.poll_once()
            tasks = store.coding_tasks(statuses=("failed",), limit=10)
        self.assertEqual(tasks[0]["task_id"], task_id)
        self.assertIn("Antigravity 失败", tasks[0]["last_summary"])
        self.assertIn("配额", tasks[0]["last_summary"])

    def test_coding_monitor_does_not_fail_running_antigravity_on_transient_auth_log(self) -> None:
        class FakeSpeech:
            def __init__(self) -> None:
                self.spoken: list[str] = []

            def speak(self, text: str, **_kwargs) -> None:
                self.spoken.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stdout_log = tmp_path / "stdout.log"
            stdout_log.write_text("", encoding="utf-8")
            executor_log = tmp_path / "antigravity.log"
            executor_log.write_text("You are not logged into Antigravity.\n", encoding="utf-8")
            store = VoiceSessionStore(tmp_path / "voice.sqlite", "test-session")
            task_id = store.create_coding_task({
                "run_id": "agy-auth-refresh",
                "pid": os.getpid(),
                "status": "running",
                "executor": "antigravity",
                "target": "测试开发",
                "stdout_log": str(stdout_log),
                "executor_log": str(executor_log),
                "next_speech_at": time.time() + 1000,
            })
            speech = FakeSpeech()
            monitor = CodingTaskMonitor(store, speech, speech_interval_seconds=60)
            monitor.poll_once()
            tasks = store.coding_tasks(statuses=("running",), limit=10)
        self.assertEqual(tasks[0]["task_id"], task_id)
        self.assertNotIn("失败", tasks[0]["last_summary"])

    def test_summarize_codex_event_extracts_key_status(self) -> None:
        self.assertEqual(
            summarize_codex_event({"type": "item.completed", "item": {"type": "command_execution", "command": "/bin/zsh -lc 'pytest'", "status": "completed"}}),
            ("running", "Codex 进展 · 运行完 pytest"),
        )
        self.assertEqual(
            summarize_codex_event({"type": "turn.completed"}),
            ("artifact_ready", "Codex 产物完成"),
        )
        self.assertEqual(
            summarize_coding_event({"type": "turn.completed"}, "antigravity"),
            ("artifact_ready", "Antigravity 产物完成"),
        )

    def test_tool_started_event_metadata_marks_cooldown_suppression(self) -> None:
        metadata = tool_started_event_metadata(
            tool_name="web_search",
            log_word="我查查",
            action_subject="特朗普",
            spoken_text="我查查 特朗普",
            spoke_start=False,
            speech_enabled=True,
            cooldown_remaining=1.2345,
            arguments={"query": "Trump"},
            turn_id="turn-1",
        )
        self.assertEqual(metadata["speech_suppressed_reason"], "cooldown")
        self.assertEqual(metadata["speech_cooldown_remaining"], 1.234)
        self.assertEqual(metadata["turn_id"], "turn-1")

    def test_tool_started_event_metadata_omits_empty_turn_id(self) -> None:
        metadata = tool_started_event_metadata(
            tool_name="web_search",
            log_word="我查查",
            action_subject="特朗普",
            spoken_text="我查查 特朗普",
            spoke_start=True,
            speech_enabled=True,
            cooldown_remaining=0,
            arguments={},
        )
        self.assertEqual(metadata["speech_suppressed_reason"], "")
        self.assertNotIn("turn_id", metadata)

    def test_tool_voice_summary_event_metadata_marks_cooldown_suppression(self) -> None:
        metadata = tool_voice_summary_event_metadata(
            tool_name="web_search",
            ok=True,
            phrase="搜到了",
            spoken=False,
            speech_enabled=True,
            cooldown_remaining=2.3456,
            cooldown_bypassed=False,
            turn_id="turn-2",
        )
        self.assertEqual(metadata["speech_suppressed_reason"], "cooldown")
        self.assertEqual(metadata["speech_cooldown_remaining"], 2.346)
        self.assertFalse(metadata["speech_cooldown_bypassed"])
        self.assertEqual(metadata["turn_id"], "turn-2")

    def test_tool_voice_summary_event_metadata_for_silent_tool_is_minimal(self) -> None:
        metadata = tool_voice_summary_event_metadata(
            tool_name="add_context_note",
            ok=True,
            phrase="记下来了",
            spoken=False,
            speech_enabled=False,
        )
        self.assertFalse(metadata["speech_enabled"])
        self.assertNotIn("speech_suppressed_reason", metadata)
        self.assertNotIn("turn_id", metadata)

    def test_tool_retry_helpers_format_timeout_and_backoff(self) -> None:
        self.assertEqual(tool_timeout_error_message("web_search", 20, 2, 3), "tool web_search timed out after 20.0s on attempt 2/3")
        self.assertEqual(tool_retry_backoff_seconds(1), 0.8)
        self.assertEqual(tool_retry_backoff_seconds(99), 2.0)

    def test_tool_retry_event_metadata_truncates_reason_and_keeps_timeout(self) -> None:
        metadata = tool_retry_event_metadata(attempt=1, attempts=3, reason="x" * 1200, timeout_seconds=20.0)
        self.assertEqual(metadata["attempt"], 1)
        self.assertEqual(metadata["attempts"], 3)
        self.assertEqual(metadata["timeout_seconds"], 20.0)
        self.assertEqual(len(metadata["reason"]), 1000)

    def test_plan_prefetch_started_and_completed_metadata(self) -> None:
        state = {
            "tool_name": "web_search",
            "arguments": {"query": "Trump"},
            "ok": True,
            "error": None,
            "elapsed_seconds": 0.123,
        }
        self.assertEqual(
            plan_prefetch_started_metadata([state], turn_id="turn-1"),
            {"count": 1, "tools": ["web_search"], "turn_id": "turn-1"},
        )
        self.assertEqual(
            plan_prefetch_completed_metadata(state, turn_id="turn-1"),
            {
                "tool_name": "web_search",
                "arguments": {"query": "Trump"},
                "ok": True,
                "error": None,
                "elapsed_seconds": 0.123,
                "turn_id": "turn-1",
            },
        )

    def test_plan_prefetch_hit_and_miss_metadata(self) -> None:
        state = {"elapsed_seconds": 0.5}
        arguments = {"query": "Trump"}
        self.assertEqual(plan_prefetch_hit_metadata(state, arguments), {"arguments": arguments, "elapsed_seconds": 0.5})
        self.assertEqual(
            plan_prefetch_miss_metadata(reason="prefetch still running", wait_seconds=0.25, arguments=arguments),
            {"reason": "prefetch still running", "arguments": arguments, "wait_seconds": 0.25},
        )
        self.assertEqual(
            plan_prefetch_miss_metadata(reason="prefetch failed", error="boom", arguments=arguments),
            {"reason": "prefetch failed", "arguments": arguments, "error": "boom"},
        )

    def test_agno_tool_output_metadata_is_stable(self) -> None:
        metadata = agno_tool_output_metadata(
            tool_name="web_search",
            log_word="搜到了",
            ok=True,
            arguments={"query": "Trump"},
            result={"results": []},
            summary="搜到了",
        )
        self.assertEqual(
            metadata,
            {
                "tool_name": "web_search",
                "log_word": "搜到了",
                "log_language": "zh",
                "ok": True,
                "arguments": {"query": "Trump"},
                "result": {"results": []},
                "summary": "搜到了",
            },
        )


class RuntimeToolPolicyTests(unittest.TestCase):
    def test_open_url_disables_fullscreen_when_arranging_workspace(self) -> None:
        decision = prepare_runtime_tool_call(
            "open_url_in_browser",
            {"url": "https://youtube.com/watch?v=abc", "fullscreen": True, "video_fullscreen": True},
            "打开视频并把 Chrome 和 Codex 并排",
            None,
        )
        self.assertFalse(decision.arguments["fullscreen"])
        self.assertFalse(decision.arguments["video_fullscreen"])
        self.assertIsNone(decision.blocked_payload)

    def test_arrange_workspace_arguments_are_normalized_from_user_text(self) -> None:
        decision = prepare_runtime_tool_call("arrange_workspace", {}, "把 Chrome 和 Codex 并排", None)
        self.assertEqual(decision.arguments["mode"], "parallel")
        self.assertIn("Google Chrome", decision.arguments["app_names"])
        self.assertIn("Codex", decision.arguments["app_names"])

    def test_manual_window_layout_osascript_is_blocked(self) -> None:
        decision = prepare_runtime_tool_call(
            "run_osascript",
            {"script": 'tell application "System Events" to set bounds of window 1 to {0, 0, 500, 500}'},
            "把 Chrome 和 Codex 排窗口",
            None,
        )
        self.assertEqual(decision.blocked_reason, "manual window layout blocked")
        self.assertEqual(decision.blocked_payload["error"], "window layout must use arrange_workspace")
        self.assertIn("suggested_arguments", decision.blocked_payload)

    def test_preflight_suppresses_duplicate_followup(self) -> None:
        state = ToolTurnState()
        first = evaluate_runtime_tool_preflight("trigger_fast_followup", {"prompt": "我查到了"}, "", None, state)
        second = evaluate_runtime_tool_preflight("trigger_fast_followup", {"prompt": " 我查到了。 "}, "", None, state)
        self.assertEqual(first.action, "allow")
        self.assertEqual(second.action, "suppress")
        self.assertEqual(second.event_kind, "tool_duplicate_followup_suppressed")

    def test_preflight_blocks_missing_plan_requirement(self) -> None:
        state = ToolTurnState()
        plan = {
            "steps": [
                {"kind": "tool", "order": 1, "suggested_tools": ["web_search"]},
                {"kind": "tool", "order": 2, "suggested_tools": ["open_url_in_browser"]},
            ]
        }
        decision = evaluate_runtime_tool_preflight("open_url_in_browser", {"url": "https://example.com"}, "打开网页", plan, state)
        self.assertEqual(decision.action, "block")
        self.assertEqual(decision.blocked_reason, "local action not completed")
        self.assertEqual(decision.blocked_payload["missing_tools"], ["web_search"])

    def test_preflight_suppresses_duplicate_video_open(self) -> None:
        state = ToolTurnState()
        state.opened_video_urls.add("https://youtube.com/watch?v=abc")
        decision = evaluate_runtime_tool_preflight(
            "open_url_in_browser",
            {"url": "https://youtube.com/watch?v=def"},
            "打开一个 YouTube 视频",
            None,
            state,
        )
        self.assertEqual(decision.action, "suppress")
        self.assertEqual(decision.event_kind, "tool_duplicate_open_suppressed")

    def test_preflight_blocks_camera_open_as_snapshot(self) -> None:
        state = ToolTurnState()
        decision = evaluate_runtime_tool_preflight("capture_camera_snapshot", {}, "打开 camera 应用", None, state)
        self.assertEqual(decision.action, "block")
        self.assertTrue(decision.queue_voice_summary)
        self.assertIn("not camera snapshot", decision.blocked_payload["error"])

    def test_preflight_blocks_unverified_video_url(self) -> None:
        state = ToolTurnState()
        decision = evaluate_runtime_tool_preflight(
            "open_url_in_browser",
            {"url": "https://youtube.com/watch?v=madeup"},
            "播放一个视频",
            None,
            state,
        )
        self.assertEqual(decision.action, "block")
        self.assertEqual(decision.blocked_payload["error"], "video URL not verified in current turn")


class ToolTurnStateTests(unittest.TestCase):
    def test_register_followup_text_dedupes_normalized_text(self) -> None:
        state = ToolTurnState()
        self.assertFalse(state.register_followup_text(" 我查到了。 "))
        self.assertTrue(state.register_followup_text("我查到了"))

    def test_failed_signature_tracking(self) -> None:
        state = ToolTurnState()
        self.assertFalse(state.has_failed_signature("abc"))
        state.mark_failed_signature("abc")
        self.assertTrue(state.has_failed_signature("abc"))

    def test_mark_success_tracks_tool_state(self) -> None:
        state = ToolTurnState()
        result = {"url": "https://youtube.com/watch?v=abc", "results": [{"title": "视频", "url": "https://youtube.com/watch?v=abc"}]}
        state.mark_success("open_url_in_browser", "sig", {"url": "https://youtube.com/watch?v=abc"}, result)
        self.assertIn("open_url_in_browser", state.completed_ok_tools)
        self.assertIn("sig", state.completed_action_signatures)
        self.assertIn("https://youtube.com/watch?v=abc", state.opened_video_urls)


if __name__ == "__main__":
    unittest.main()
