from __future__ import annotations

import html
import json
from typing import Any

from .front_note import (
    front_note_markdown_to_html,
    render_front_note_media_cards,
    sanitize_front_note_html,
)


def live_log_source_group(item: dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return "agent"
    role = str(item.get("role") or "")
    lane = str(item.get("lane") or "")
    if role == "user":
        return "user"
    return "agent"


def render_front_note_document(state: dict[str, Any]) -> str:
    live = state.get("live") or {}
    content_html = sanitize_front_note_html(str(live.get("html") or "")) or front_note_markdown_to_html(str(state.get("content") or ""))
    media_html = render_front_note_media_cards(live.get("media") or state.get("media") or [])
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{ color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    html, body {{ margin: 0; min-height: 100%; background: transparent; color: #171814; }}
    body {{ box-sizing: border-box; padding: 14px; overflow: hidden; }}
    .card {{ min-height: calc(100vh - 28px); box-sizing: border-box; padding: 18px 18px 16px; border-radius: 18px; background: rgba(252, 250, 244, .92); border: 1px solid rgba(42,42,36,.12); box-shadow: 0 18px 48px rgba(24,24,20,.18); backdrop-filter: blur(24px) saturate(1.18); overflow: auto; }}
    .handle {{ width: 42px; height: 4px; border-radius: 999px; background: rgba(35,35,31,.22); margin: 0 auto 12px; }}
    h1, h2, h3 {{ margin: 10px 0 8px; font-weight: 700; letter-spacing: 0; }}
    h1 {{ font-size: 20px; }} h2 {{ font-size: 17px; }} h3 {{ font-size: 15px; }}
    p, li {{ font-size: 15px; line-height: 1.48; margin: 8px 0; word-break: break-word; }}
    ul {{ padding-left: 20px; margin: 8px 0; }}
    .empty {{ color: rgba(24,24,20,.48); }}
    figure {{ margin: 12px 0 0; }}
    img {{ display: block; max-width: 100%; border-radius: 12px; }}
    figcaption {{ margin-top: 6px; font-size: 12px; color: rgba(24,24,20,.58); }}
    .link-card {{ display: block; margin-top: 10px; padding: 11px 12px; border-radius: 12px; border: 1px solid rgba(42,42,36,.12); color: #171814; text-decoration: none; background: rgba(255,255,255,.54); }}
    .link-card span {{ display: block; font-size: 14px; font-weight: 650; }}
    .link-card small {{ display: block; margin-top: 4px; color: rgba(24,24,20,.55); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  </style>
</head>
<body>
  <div class="card"><div class="handle"></div>{content_html}{media_html}</div>
</body>
</html>"""


def render_front_note_editor_document(api_url: str) -> str:
    api_json = json.dumps(api_url, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    html, body {{ margin: 0; min-height: 100%; background: transparent; color: #171814; overflow: hidden; }}
    body {{ box-sizing: border-box; padding: 12px; }}
    .shell {{ position: relative; height: calc(100vh - 24px); box-sizing: border-box; display: grid; grid-template-rows: auto minmax(0, 1fr); border-radius: 20px; background: rgba(252,250,244,.88); border: 1px solid rgba(34,34,28,.12); box-shadow: 0 18px 48px rgba(24,24,20,.18); backdrop-filter: blur(28px) saturate(1.18); overflow: hidden; }}
    .topbar {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; padding: 10px 11px 8px; border-bottom: 1px solid rgba(34,34,28,.08); }}
    .tabs {{ display: flex; gap: 6px; align-items: center; min-width: 0; }}
    .tab {{ appearance: none; border: 0; border-radius: 10px; padding: 7px 11px; background: transparent; color: rgba(24,24,20,.58); font: 650 13px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; cursor: default; }}
    .tab.active {{ color: #171814; background: rgba(255,255,255,.70); box-shadow: inset 0 0 0 1px rgba(34,34,28,.10); }}
    .tools {{ display: flex; gap: 5px; justify-content: flex-end; align-items: center; min-width: 0; }}
    .tool {{ border: 1px solid rgba(34,34,28,.12); background: rgba(255,255,255,.56); border-radius: 9px; height: 28px; padding: 0 8px; font: 650 12px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: rgba(24,24,20,.72); white-space: nowrap; }}
    .status {{ padding: 0 2px; font-size: 11px; color: rgba(24,24,20,.46); white-space: nowrap; }}
    .pending {{ display: none; align-items: center; gap: 7px; padding: 6px 11px; border-bottom: 1px solid rgba(147,109,26,.18); background: rgba(255,236,176,.55); color: #4a3510; font-size: 12px; }}
    .pending.show {{ display: flex; }}
    .pending button {{ border: 1px solid rgba(74,53,16,.14); background: rgba(255,255,255,.62); border-radius: 8px; height: 24px; padding: 0 8px; color: #4a3510; font: 650 12px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .live-panel {{ position: relative; min-height: 0; height: 100%; display: none; grid-template-rows: minmax(0, 1fr) auto; overflow: hidden; }}
    .live-top-controls {{ display: none; align-items: center; justify-content: flex-end; gap: 6px; min-width: 0; max-width: min(58vw, 360px); }}
    .live-count-toggle {{ height: 28px; border: 1px solid rgba(34,34,28,.12); border-radius: 999px; background: rgba(255,255,255,.56); color: rgba(24,24,20,.66); padding: 0 9px; font: 700 11px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; white-space: nowrap; }}
    .live-count-toggle.active {{ color: #171814; background: rgba(255,255,255,.82); box-shadow: inset 0 0 0 1px rgba(34,34,28,.06); }}
    .live-menu {{ position: absolute; top: 48px; left: 0; right: 0; z-index: 20; display: flex; flex-wrap: wrap; gap: 4px; align-content: flex-start; max-height: min(260px, calc(100% - 58px)); overflow: auto; opacity: 0; visibility: hidden; pointer-events: none; transform: translateY(-7px) scaleY(.88); transform-origin: top center; padding: 6px; border: 1px solid rgba(34,34,28,0); border-radius: 0 0 16px 16px; background: rgba(252,250,244,.94); box-shadow: 0 16px 42px rgba(24,24,20,0); backdrop-filter: blur(18px) saturate(1.08); transition: opacity 90ms ease-out, transform 120ms ease-out, visibility 0s linear 120ms, border-color 120ms ease-out, box-shadow 120ms ease-out; }}
    .live-menu.show {{ opacity: 1; visibility: visible; pointer-events: auto; transform: translateY(0) scaleY(1); border-color: rgba(34,34,28,.12); box-shadow: 0 16px 42px rgba(24,24,20,.18); transition-delay: 0s; }}
    .live-session-option {{ display: inline-flex; align-items: center; max-width: 100%; gap: 5px; border: 1px solid rgba(34,34,28,.09); border-radius: 999px; background: rgba(255,255,255,.48); color: rgba(24,24,20,.68); padding: 4px 7px; text-align: left; font: 700 10px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .live-session-option.active {{ border-color: rgba(34,34,28,.16); background: rgba(255,255,255,.82); color: #171814; }}
    .live-session-name {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 180px; }}
    .live-session-count {{ color: rgba(24,24,20,.42); white-space: nowrap; }}
    .live-meta {{ font-size: 11px; color: rgba(24,24,20,.46); white-space: nowrap; }}
    .live-stats {{ color: rgba(24,24,20,.64); font: 10px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .live-stat {{ display: inline-flex; margin: 2px; border: 1px solid rgba(34,34,28,.09); border-radius: 999px; padding: 4px 7px; background: rgba(255,255,255,.48); }}
    .live-log {{ min-height: 0; overflow: auto; padding: 10px 13px 12px; background: transparent; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11px; line-height: 1.42; }}
    .live-log * {{ background-color: transparent; }}
    .log-row {{ display: grid; grid-template-columns: 84px minmax(0, 1fr); gap: 12px; align-items: baseline; padding: 3px 0; background: transparent; border: 0; border-radius: 0; box-shadow: none; color: rgba(24,24,20,.82); }}
    .log-mark {{ display: flex; align-items: baseline; min-width: 0; white-space: nowrap; color: rgba(24,24,20,.38); }}
    .log-role-row {{ display: grid; grid-template-columns: 84px minmax(0, 1fr); gap: 12px; margin-top: 10px; padding: 0; background: transparent; border: 0; border-radius: 0; box-shadow: none; color: rgba(24,24,20,.52); }}
    .log-role-row:first-child {{ margin-top: 0; }}
    .log-role-mark {{ display: flex; align-items: center; gap: 5px; white-space: nowrap; }}
    .log-role-mark::before {{ content: ""; width: 5px; height: 5px; border-radius: 999px; background: rgba(24,24,20,.22); }}
    .log-role-row.user .log-role-mark::before {{ background: rgba(40,100,216,.78); }}
    .log-role-row.agent .log-role-mark::before {{ background: rgba(39,143,88,.78); }}
    .log-role-row.tool .log-role-mark::before {{ background: rgba(123,97,217,.78); }}
    .log-role-row.system .log-role-mark::before {{ background: rgba(24,24,20,.28); }}
    .log-role-row.error .log-role-mark::before {{ background: rgba(200,58,50,.78); }}
    .log-role {{ font: 820 12px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: .035em; }}
    .log-time {{ color: rgba(24,24,20,.38); white-space: nowrap; font-variant-numeric: tabular-nums; }}
    .log-text {{ min-width: 0; white-space: pre-wrap; word-break: break-word; }}
    .log-more {{ appearance: none; border: 0; padding: 0 0 0 4px; background: transparent; color: rgba(40,100,216,.92); font: 700 11px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; cursor: default; }}
    .live-compose {{ display: grid; grid-template-columns: minmax(0, 1fr) 72px; gap: 8px; align-items: end; padding: 9px 12px 12px; border-top: 1px solid rgba(34,34,28,.08); }}
    .live-compose textarea {{ resize: none; box-sizing: border-box; width: 100%; height: 38px; min-height: 38px; max-height: 86px; border: 1px solid rgba(34,34,28,.12); border-radius: 11px; background: rgba(255,255,255,.62); color: #171814; padding: 9px 10px; font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; outline: none; }}
    .live-compose button {{ width: 72px; height: 38px; align-self: end; border: 0; border-radius: 11px; background: #171814; color: #fff; padding: 0; font: 700 12px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .editor-wrap {{ min-height: 0; height: 100%; overflow: auto; padding: 14px 16px 12px; }}
    #editor {{ min-height: 260px; outline: none; font-size: 16px; line-height: 1.52; word-break: break-word; white-space: normal; }}
    #editor:empty::before {{ content: "写点东西..."; color: rgba(24,24,20,.38); }}
    #editor p, #editor li {{ margin: 8px 0; }}
    #editor h1 {{ font-size: 20px; }} #editor h2 {{ font-size: 17px; }} #editor h3 {{ font-size: 15px; }}
    #editor img {{ max-width: 100%; border-radius: 12px; display: block; margin: 9px 0; }}
    .attachments {{ display: grid; gap: 6px; padding: 0 12px 12px; max-height: 106px; overflow: auto; }}
    .attachment, .link-card {{ display: block; padding: 9px 10px; border-radius: 11px; border: 1px solid rgba(42,42,36,.12); color: #171814; text-decoration: none; background: rgba(255,255,255,.56); }}
    .attachment strong, .link-card span {{ display: block; font-size: 13px; font-weight: 650; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .attachment small, .link-card small {{ display: block; margin-top: 4px; color: rgba(24,24,20,.55); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    audio {{ display: block; width: 100%; margin-top: 6px; height: 28px; }}
    input[type=file] {{ display: none; }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div class="tabs">
        <button class="tab" data-tab="live">Live</button>
        <button class="tab" data-tab="context">Context</button>
      </div>
      <div class="tools">
        <div class="live-top-controls" id="liveTopControls">
          <button class="live-count-toggle" id="sessionTag" type="button" title="Session">Session</button>
          <button class="live-count-toggle" id="liveStatsToggle" type="button">计数</button>
          <span class="live-meta" id="liveMeta"></span>
        </div>
        <span class="status" id="noteStatus"></span>
        <button class="tool" id="bold" title="加粗">加粗</button>
        <button class="tool" id="link" title="插入链接">链接</button>
        <button class="tool" id="file" title="添加图片、音频或文件">附件</button>
        <button class="tool" id="clear" title="清空当前 tab">清空</button>
        <input id="fileInput" type="file" multiple>
      </div>
    </div>
    <div class="live-menu" id="sessionMenu"></div>
    <div class="live-menu live-stats" id="liveStats"></div>
    <div class="pending" id="pendingBar">
      <span id="pendingText"></span>
      <button id="mergePending" type="button">合并</button>
      <button id="replacePending" type="button">替换</button>
    </div>
    <div class="live-panel" id="livePanel">
      <div class="live-log" id="liveLog"></div>
      <div class="live-compose">
        <textarea id="liveNoteInput" placeholder="输入 prompt"></textarea>
        <button id="liveNoteSend" type="button">发送</button>
      </div>
    </div>
    <div class="editor-wrap"><div id="editor" contenteditable="true"></div></div>
    <div class="attachments" id="attachments"></div>
  </div>
  <script>
    const API = {api_json};
    const API_BASE = API.replace(/\\/api\\/front-note$/, "");
    const tabs = {{ live: {{html:"", text:"", media:[], version:0}}, context: {{html:"", text:"", media:[], version:0}} }};
    let activeTab = "live";
    let stateVersion = -1;
    let currentSessionId = "";
    let liveSessions = [];
    let liveItemsKey = "";
    let liveSessionsLoadedAt = 0;
    let liveStatsVisible = false;
    let sessionMenuVisible = false;
    let dirty = false;
    let focused = false;
    let saveTimer = null;
    let localEditingUntil = 0;
    const pendingRemote = {{ live: null, context: null }};
    const loadedTabs = {{ live: false, context: false }};
    const editor = document.getElementById("editor");
    const attachments = document.getElementById("attachments");
    const fileInput = document.getElementById("fileInput");
    const noteStatus = document.getElementById("noteStatus");
    const pendingBar = document.getElementById("pendingBar");
    const pendingText = document.getElementById("pendingText");
    const livePanel = document.getElementById("livePanel");
    const liveLog = document.getElementById("liveLog");
    const liveStats = document.getElementById("liveStats");
    const liveStatsToggle = document.getElementById("liveStatsToggle");
    const liveTopControls = document.getElementById("liveTopControls");
    const liveMeta = document.getElementById("liveMeta");
    const sessionTag = document.getElementById("sessionTag");
    const sessionMenu = document.getElementById("sessionMenu");
    const liveNoteInput = document.getElementById("liveNoteInput");
    const liveNoteSend = document.getElementById("liveNoteSend");

    function textFromHtml(value) {{
      const div = document.createElement("div");
      div.innerHTML = value || "";
      return (div.innerText || "").trim();
    }}
    function escapeHtml(value) {{
      return String(value || "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#039;"}}[ch]));
    }}
    function renderAttachments() {{
      attachments.innerHTML = "";
      for (const item of (tabs[activeTab].media || [])) {{
        const a = document.createElement(item.type === "link" || item.type === "file" ? "a" : "div");
        a.className = item.type === "link" || item.type === "file" ? "link-card" : "attachment";
        if (a.tagName === "A") {{ a.href = item.url || "#"; a.target = "_blank"; a.rel = "noreferrer"; }}
        const title = escapeHtml(item.title || item.url || "附件");
        const sub = escapeHtml(item.caption || item.url || "");
        if (item.type === "audio") {{
          a.innerHTML = `<strong>${{title}}</strong><audio controls src="${{escapeHtml(item.url || "")}}"></audio><small>${{sub}}</small>`;
        }} else {{
          a.innerHTML = `<span>${{title}}</span><small>${{sub}}</small>`;
        }}
        attachments.appendChild(a);
      }}
    }}
    function renderTabs() {{
      document.querySelectorAll(".tab").forEach(btn => btn.classList.toggle("active", btn.dataset.tab === activeTab));
    }}
    function updateStatus(extra="") {{
      if (activeTab === "live") {{
        noteStatus.textContent = extra || "Live log";
        return;
      }}
      const textLen = (tabs[activeTab]?.text || textFromHtml(tabs[activeTab]?.html || "") || "").length;
      noteStatus.textContent = `${{activeTab === "context" ? "Context" : "Live"}} · ${{textLen}}字${{extra ? " · " + extra : ""}}`;
    }}
    function isLocalLocked() {{
      return focused || dirty || Date.now() < localEditingUntil;
    }}
    function touchLocalLock() {{
      localEditingUntil = Date.now() + 4500;
    }}
    function renderPending() {{
      pendingBar.classList.remove("show");
    }}
    async function flushContextPendingIfReady() {{
      const pending = pendingRemote.context;
      if (!pending || activeTab !== "context" || isLocalLocked()) return;
      tabs.context.html = pending.html || "";
      tabs.context.text = textFromHtml(tabs.context.html);
      tabs.context.media = pending.media || [];
      tabs.context.version = Math.max(Number(tabs.context.version || 0), Number(pending.version || 0));
      pendingRemote.context = null;
      editor.innerHTML = tabs.context.html;
      renderAttachments();
      updateStatus("已追加");
      dirty = false;
    }}
    function applyTab(force=false) {{
      if (!force && focused && dirty) return;
      const isLive = activeTab === "live";
      livePanel.style.display = isLive ? "grid" : "none";
      liveTopControls.style.display = isLive ? "flex" : "none";
      document.querySelector(".editor-wrap").style.display = isLive ? "none" : "block";
      attachments.style.display = isLive ? "none" : "grid";
      document.getElementById("bold").style.display = isLive ? "none" : "";
      document.getElementById("link").style.display = isLive ? "none" : "";
      document.getElementById("file").style.display = isLive ? "none" : "";
      document.getElementById("clear").style.display = isLive ? "none" : "";
      if (isLive) {{
        renderTabs();
        updateStatus();
        loadLiveSessions(true);
        return;
      }}
      editor.innerHTML = tabs[activeTab].html || "";
      renderTabs();
      renderAttachments();
      updateStatus();
      renderPending();
      dirty = false;
    }}
    function scheduleSave() {{
      dirty = true;
      loadedTabs[activeTab] = true;
      touchLocalLock();
      clearTimeout(saveTimer);
      saveTimer = setTimeout(saveNow, 450);
    }}
    async function saveNow() {{
      if (activeTab === "live") {{
        dirty = false;
        return;
      }}
      const htmlNow = editor.innerHTML || "";
      const textNow = textFromHtml(htmlNow);
      if (!loadedTabs[activeTab] && !textNow && !(tabs[activeTab].media || []).length) {{
        dirty = false;
        updateStatus("等待同步");
        return;
      }}
      tabs[activeTab].html = htmlNow;
      tabs[activeTab].text = textNow;
      const payload = {{ action:"update", tab:activeTab, active_tab:activeTab, html:htmlNow, media:tabs[activeTab].media || [], source:"human", allow_empty:loadedTabs[activeTab] }};
      try {{
        const res = await fetch(API, {{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify(payload)}});
        const data = await res.json();
        if (data && data.state) mergeState(data.state, false);
        dirty = false;
        updateStatus("已保存");
      }} catch (err) {{}}
    }}
    function mergeState(state, apply=true) {{
      if (!state) return;
      stateVersion = Number(state.version || stateVersion || 0);
      const remoteActive = state.active_tab || activeTab;
      for (const name of ["live", "context"]) {{
        const remote = state[name] || {{}};
        if (!tabs[name] || Number(remote.version || 0) >= Number(tabs[name].version || 0)) {{
          if (name === activeTab && isLocalLocked()) {{
            if (Number(remote.version || 0) > Number(tabs[name].version || 0) && name === "context") pendingRemote[name] = {{html: remote.html || "", text: remote.text || "", media: remote.media || [], version: Number(remote.version || 0)}};
          }} else {{
            tabs[name] = {{html: remote.html || "", text: remote.text || "", media: remote.media || [], version: Number(remote.version || 0)}};
            loadedTabs[name] = true;
            if (pendingRemote[name] && Number(pendingRemote[name].version || 0) <= Number(tabs[name].version || 0)) pendingRemote[name] = null;
          }}
        }}
      }}
      if (!focused || !dirty) activeTab = remoteActive;
      if (apply) applyTab(false);
      updateStatus();
      renderPending();
      flushContextPendingIfReady();
    }}
    window.__frontNoteApplyState = function(state) {{
      mergeState(state, true);
    }};
    async function poll() {{
      try {{
        const res = await fetch(API, {{cache:"no-store"}});
        const state = await res.json();
        if (Number(state.version || 0) !== stateVersion) mergeState(state, true);
      }} catch (err) {{}}
    }}
    function liveGroup(item) {{
      if (!item) return "agent";
      if (item.role === "user") return "user";
      return "agent";
    }}
    function liveGroupLabel(group) {{
      return ({{user:"USER", agent:"AGENT"}})[group] || "AGENT";
    }}
    function itemText(item, full=false) {{
      if (!item) return "";
      if (item.kind === "tool_event") {{
        const meta = item.metadata || {{}};
        const result = meta.result || {{}};
        const summary = result._summary || result.summary || result.result || "";
        const head = `${{meta.tool_name || item.content || "tool"}} ${{meta.ok ? "✓" : "×"}}${{summary ? " · " + summary : ""}}`;
        if (!full) return head;
        const details = [
          `args: ${{stableJson(meta.arguments || {{}})}}`,
          `result: ${{stableJson(result || {{}})}}`
        ].join("\\n");
        return `${{head}}\\n${{details}}`;
      }}
      if (item.kind === "timing") {{
        const meta = item.metadata || {{}};
        const dur = meta.duration_seconds == null ? "" : ` · ${{meta.duration_seconds}}s`;
        return `${{meta.label || item.content || meta.stage || "timing"}}${{dur}}`;
      }}
      if (item.kind === "llm_call") {{
        const meta = item.metadata || {{}};
        const dur = meta.duration_seconds == null ? "" : ` · ${{meta.duration_seconds}}s`;
        return `LLM · ${{meta.phase || "model"}} ${{meta.model || item.lane || ""}} ${{meta.status || ""}}${{dur}}`;
      }}
      if (item.kind === "task_counts") {{
        const meta = item.metadata || {{}};
        return `counts · ps=${{meta.tr ?? 0}} tc=${{meta.tc ?? 0}} mc=${{meta.mc ?? 0}} tts=${{meta.tts ?? 0}}`;
      }}
      if (item.kind === "domain_probe") {{
        try {{
          const payload = JSON.parse(item.content || "{{}}");
          const domains = payload.domains || [];
          if (!domains.length) return "probe · no domain";
          const summary = "probe · " + domains.map(domain => {{
            const calls = (domain.suggested_actions || []).map(action => action.tool_call).filter(Boolean);
            const call = calls.length ? " → " + calls.join(" / ") : "";
            return `${{domain.domain || "domain"}}:${{domain.intent || ""}}${{call}}`;
          }}).join(" / ");
          return full ? `${{summary}}\\n${{stableJson(payload)}}` : summary;
        }} catch (err) {{
          return item.content || "probe";
        }}
      }}
      if (item.kind === "task_log") {{
        const meta = item.metadata || {{}};
        if (item.content && item.content.includes("任务结束") && meta.registered_tool_count != null) {{
          return `${{item.content}} · ps=${{meta.registered_tool_count ?? 0}} tc=${{meta.tool_count ?? 0}} mc=${{meta.model_call_count ?? 0}} tts=${{meta.tts_call_count ?? 0}}`;
        }}
        return (item.content || "task").replace(/ · tools×\\d+/, "");
      }}
      const base = item.content || item.kind || "";
      if (!full) return base;
      const meta = item.metadata && Object.keys(item.metadata || {{}}).length ? `\\nmeta: ${{stableJson(item.metadata)}}` : "";
      return `${{base}}${{meta}}`;
    }}
    function stableJson(value) {{
      try {{
        return JSON.stringify(value, null, 2);
      }} catch (err) {{
        return String(value || "");
      }}
    }}
    function compactLogText(text, limit=360) {{
      const value = String(text || "");
      const lines = value.split("\\n");
      if (value.length <= limit && lines.length <= 6) return {{display:value, truncated:false}};
      let display = value.slice(0, limit);
      const lineDisplay = lines.slice(0, 6).join("\\n");
      if (lineDisplay.length < display.length) display = lineDisplay;
      return {{display:display.replace(/\\s+$/,""), truncated:true}};
    }}
    function timeLabel(ts) {{
      if (!ts) return "";
      const d = new Date(Number(ts) * 1000);
      return d.toLocaleTimeString([], {{hour:"2-digit", minute:"2-digit", second:"2-digit", hour12:false}});
    }}
    function countSummary(rows, prefix) {{
      return (rows || []).slice(0, 5).map(item => `<span class="live-stat">${{escapeHtml(prefix)}}:${{escapeHtml(item.name)}}×${{escapeHtml(item.count)}}</span>`).join("");
    }}
    function renderLiveStats(stats) {{
      const html = [
        countSummary(stats?.calls, "call"),
        countSummary(stats?.lanes, "lane"),
        countSummary(stats?.domains, "domain"),
        countSummary(stats?.tools, "tool"),
      ].filter(Boolean).join("");
      liveStats.innerHTML = html || '<span class="live-stat">no stats</span>';
    }}
    function syncLiveStatsToggle() {{
      liveStats.classList.toggle("show", liveStatsVisible);
      liveStatsToggle.classList.toggle("active", liveStatsVisible);
      liveStatsToggle.textContent = "计数";
    }}
    function renderSessionControls() {{
      const current = liveSessions.find(session => session.session_id === currentSessionId) || {{}};
      sessionTag.textContent = "Session";
      sessionTag.title = `${{currentSessionId || "Session"}}${{current.turn_count == null ? "" : " · " + current.turn_count + " turns"}}`;
      sessionMenu.innerHTML = "";
      if (!liveSessions.length) {{
        sessionMenu.innerHTML = '<span class="live-stat">loading</span>';
      }}
      for (const session of liveSessions) {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = `live-session-option ${{session.session_id === currentSessionId ? "active" : ""}}`;
        const rawName = String(session.session_id || "");
        const shortName = rawName.length > 18 ? `${{rawName.slice(0, 7)}}...${{rawName.slice(-7)}}` : rawName;
        button.title = `${{rawName}} · ${{session.turn_count || 0}} turns`;
        button.innerHTML = `<span class="live-session-name">${{escapeHtml(session.current ? "• " : "")}}${{escapeHtml(shortName)}}</span><span class="live-session-count">${{escapeHtml(session.turn_count || 0)}}</span>`;
        button.onclick = () => {{
          currentSessionId = session.session_id || "";
          sessionMenuVisible = false;
          syncSessionMenu();
          liveItemsKey = "";
          renderSessionControls();
          loadLiveSession();
        }};
        sessionMenu.appendChild(button);
      }}
      syncSessionMenu();
    }}
    function syncSessionMenu() {{
      sessionMenu.classList.toggle("show", sessionMenuVisible);
      sessionTag.classList.toggle("active", sessionMenuVisible);
    }}
    async function loadLiveSessions(force=false) {{
      if (activeTab !== "live") return;
      if (!force && currentSessionId && Date.now() - liveSessionsLoadedAt < 10000) {{
        await loadLiveSession();
        return;
      }}
      try {{
        liveSessionsLoadedAt = Date.now();
        const res = await fetch(`${{API_BASE}}/api/live-sessions?limit=50`, {{cache:"no-store"}});
        const data = await res.json();
        const previous = currentSessionId || data.current_session_id || "";
        liveSessions = data.sessions || [];
        currentSessionId = liveSessions.some(session => session.session_id === previous) ? previous : (data.current_session_id || liveSessions[0]?.session_id || "");
        renderSessionControls();
        await loadLiveSession();
      }} catch (err) {{
        liveMeta.textContent = "log unavailable";
      }}
    }}
    async function loadLiveSession() {{
      if (activeTab !== "live" || !currentSessionId) return;
      const wasBottom = liveLog.scrollTop + liveLog.clientHeight >= liveLog.scrollHeight - 8;
      try {{
        const res = await fetch(`${{API_BASE}}/api/live-session?session_id=${{encodeURIComponent(currentSessionId)}}&limit=300`, {{cache:"no-store"}});
        const data = await res.json();
        const items = data.items || [];
        const statsKey = JSON.stringify(data.stats || {{}});
        const key = `${{data.session_id}}:${{items.length}}:${{items.at(-1)?.source || ""}}:${{items.at(-1)?.id || ""}}:${{statsKey}}`;
        liveMeta.textContent = `${{items.length}} lines`;
        if (key === liveItemsKey) return;
        liveItemsKey = key;
        renderLiveStats(data.stats || {{}});
        liveLog.innerHTML = "";
        let lastGroup = "";
        for (const item of items) {{
          const group = liveGroup(item);
          const switched = group !== lastGroup;
          if (switched) lastGroup = group;
          if (switched) {{
            const role = document.createElement("div");
            role.className = `log-role-row ${{group}}`;
            role.innerHTML = `<span class="log-role-mark"><span class="log-role">${{escapeHtml(liveGroupLabel(group))}}</span></span><span></span>`;
            liveLog.appendChild(role);
          }}
          const row = document.createElement("div");
          row.className = `log-row ${{group}}`;
          const summaryText = itemText(item, false);
          const fullText = itemText(item, true);
          const packed = compactLogText(summaryText);
          const hasHiddenDetails = fullText !== summaryText || packed.truncated;
          const textId = `logtext-${{item.source || "src"}}-${{item.id || Math.random().toString(16).slice(2)}}`;
          row.innerHTML = `<span class="log-mark"><span class="log-time">${{escapeHtml(timeLabel(item.created_at))}}</span></span><span class="log-text" id="${{escapeHtml(textId)}}">${{escapeHtml(packed.display)}}${{hasHiddenDetails ? '<button class="log-more" type="button">...更多</button>' : ''}}</span>`;
          if (hasHiddenDetails) {{
            const target = row.querySelector(".log-text");
            const button = row.querySelector(".log-more");
            button.onclick = () => {{
              target.textContent = fullText;
            }};
          }}
          liveLog.appendChild(row);
        }}
        if (wasBottom) liveLog.scrollTop = liveLog.scrollHeight;
      }} catch (err) {{
        liveMeta.textContent = "log failed";
      }}
    }}
    async function sendLivePrompt() {{
      const content = liveNoteInput.value.trim();
      if (!content || !currentSessionId) return;
      liveNoteInput.value = "";
      autosizeLiveNote();
      await fetch(`${{API_BASE}}/api/text-input`, {{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify({{text:content, mode:"quality", source:"front_note"}})}});
      liveItemsKey = "";
      await loadLiveSession();
      liveLog.scrollTop = liveLog.scrollHeight;
    }}
    function autosizeLiveNote() {{
      liveNoteInput.style.height = "38px";
      liveNoteInput.style.height = `${{Math.min(86, Math.max(38, liveNoteInput.scrollHeight))}}px`;
    }}
    async function syncActiveTab(tabName) {{
      try {{
        const res = await fetch(API, {{
          method:"POST",
          headers:{{"Content-Type":"application/json"}},
          body:JSON.stringify({{action:"show", active_tab:tabName, tab:tabName, visible:true}})
        }});
        const data = await res.json();
        if (data && data.state) {{
          mergeState(data.state, true);
          activeTab = tabName;
          applyTab(true);
        }}
      }} catch (err) {{
        poll();
      }}
    }}
    function addMedia(item) {{
      tabs[activeTab].media = [...(tabs[activeTab].media || []), item].slice(0, 8);
      renderAttachments();
      scheduleSave();
    }}
    function readFile(file) {{
      const reader = new FileReader();
      reader.onload = () => {{
        const url = String(reader.result || "");
        if (file.type.startsWith("image/")) {{
          document.execCommand("insertHTML", false, `<img src="${{url}}" alt="${{escapeHtml(file.name)}}">`);
        }} else {{
          const type = file.type.startsWith("audio/") ? "audio" : "file";
          addMedia({{type, url, title:file.name, caption:file.type || "file"}});
        }}
        scheduleSave();
      }};
      reader.readAsDataURL(file);
    }}
    function selectEditorContents() {{
      editor.focus();
      const range = document.createRange();
      range.selectNodeContents(editor);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      touchLocalLock();
    }}
    function selectLiveLogContents() {{
      const range = document.createRange();
      range.selectNodeContents(liveLog);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    }}
    async function writeClipboardText(text) {{
      if (navigator.clipboard?.writeText) {{
        await navigator.clipboard.writeText(text);
        return true;
      }}
      return document.execCommand("copy");
    }}
    async function readClipboardText() {{
      if (navigator.clipboard?.readText) return await navigator.clipboard.readText();
      return "";
    }}
    function selectedTextFromInput(input) {{
      return input.value.slice(input.selectionStart || 0, input.selectionEnd || 0);
    }}
    function openLinkPrompt() {{
      const url = prompt("链接地址");
      if (!url) return;
      const title = prompt("显示文字") || url;
      addMedia({{type:"link", url, title}});
    }}
    document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", async () => {{
      if (dirty) await saveNow();
      activeTab = btn.dataset.tab || "live";
      applyTab(true);
      syncActiveTab(activeTab);
    }}));
    sessionTag.addEventListener("click", event => {{
      event.stopPropagation();
      sessionMenuVisible = !sessionMenuVisible;
      liveStatsVisible = false;
      syncLiveStatsToggle();
      renderSessionControls();
      if (sessionMenuVisible && !liveSessions.length) loadLiveSessions(true);
    }});
    document.addEventListener("click", event => {{
      if (!sessionMenu.contains(event.target) && !liveStats.contains(event.target) && event.target !== sessionTag && event.target !== liveStatsToggle) {{
        sessionMenuVisible = false;
        syncSessionMenu();
        liveStatsVisible = false;
        syncLiveStatsToggle();
      }}
    }});
    liveNoteSend.onclick = sendLivePrompt;
    liveStatsToggle.onclick = event => {{
      event.stopPropagation();
      liveStatsVisible = !liveStatsVisible;
      sessionMenuVisible = false;
      syncSessionMenu();
      syncLiveStatsToggle();
    }};
    syncLiveStatsToggle();
    liveNoteInput.addEventListener("keydown", async event => {{
      const mod = event.metaKey || event.ctrlKey;
      const key = String(event.key || "").toLowerCase();
      if (mod && key === "a") {{
        event.preventDefault();
        liveNoteInput.select();
        return;
      }}
      if (mod && key === "c") {{
        const text = selectedTextFromInput(liveNoteInput);
        if (text) {{
          event.preventDefault();
          await writeClipboardText(text);
        }}
        return;
      }}
      if (mod && key === "x") {{
        const start = liveNoteInput.selectionStart || 0;
        const end = liveNoteInput.selectionEnd || 0;
        const text = liveNoteInput.value.slice(start, end);
        if (text) {{
          event.preventDefault();
          await writeClipboardText(text);
          liveNoteInput.setRangeText("", start, end, "start");
        }}
        return;
      }}
      if (mod && key === "v") {{
        try {{
          const text = await readClipboardText();
          if (text) {{
            event.preventDefault();
            liveNoteInput.setRangeText(text, liveNoteInput.selectionStart || 0, liveNoteInput.selectionEnd || 0, "end");
          }}
        }} catch (err) {{}}
        return;
      }}
      if (mod && key === "z") {{
        document.execCommand(event.shiftKey ? "redo" : "undo");
        return;
      }}
      if (event.key === "Enter" && !event.shiftKey) {{
        event.preventDefault();
        sendLivePrompt();
      }}
    }});
    liveNoteInput.addEventListener("input", autosizeLiveNote);
    document.addEventListener("keydown", async event => {{
      if (activeTab !== "live" || document.activeElement === liveNoteInput) return;
      const mod = event.metaKey || event.ctrlKey;
      if (!mod) return;
      const key = String(event.key || "").toLowerCase();
      if (key === "a") {{
        event.preventDefault();
        selectLiveLogContents();
      }} else if (key === "c") {{
        const text = String(window.getSelection()?.toString() || "");
        if (text) {{
          event.preventDefault();
          await writeClipboardText(text);
        }}
      }} else if (key === "v") {{
        try {{
          const text = await readClipboardText();
          if (text) {{
            event.preventDefault();
            liveNoteInput.focus();
            liveNoteInput.setRangeText(text, liveNoteInput.selectionStart || 0, liveNoteInput.selectionEnd || 0, "end");
          }}
        }} catch (err) {{}}
      }}
    }});
    editor.addEventListener("focus", () => {{ focused = true; touchLocalLock(); }});
    editor.addEventListener("blur", () => {{ focused = false; localEditingUntil = Date.now() + 2500; if (dirty) saveNow(); }});
    editor.addEventListener("input", scheduleSave);
    editor.addEventListener("keydown", event => {{
      const mod = event.metaKey || event.ctrlKey;
      if (!mod) return;
      const key = String(event.key || "").toLowerCase();
      if (key === "a") {{
        event.preventDefault();
        selectEditorContents();
      }} else if (key === "b") {{
        event.preventDefault();
        document.execCommand("bold");
        scheduleSave();
      }} else if (key === "i") {{
        event.preventDefault();
        document.execCommand("italic");
        scheduleSave();
      }} else if (key === "u") {{
        event.preventDefault();
        document.execCommand("underline");
        scheduleSave();
      }} else if (key === "k") {{
        event.preventDefault();
        openLinkPrompt();
      }} else if (key === "z" && event.shiftKey) {{
        event.preventDefault();
        document.execCommand("redo");
        scheduleSave();
      }} else if (key === "z") {{
        event.preventDefault();
        document.execCommand("undo");
        scheduleSave();
      }}
    }});
    editor.addEventListener("paste", event => {{
      for (const item of event.clipboardData?.items || []) {{
        const file = item.getAsFile?.();
        if (file) readFile(file);
      }}
    }});
    editor.addEventListener("dragover", event => event.preventDefault());
    editor.addEventListener("drop", event => {{
      event.preventDefault();
      for (const file of event.dataTransfer?.files || []) readFile(file);
    }});
    document.getElementById("bold").onclick = () => {{ document.execCommand("bold"); scheduleSave(); }};
    document.getElementById("file").onclick = () => fileInput.click();
    document.getElementById("clear").onclick = async () => {{
      editor.innerHTML = "";
      tabs[activeTab].html = "";
      tabs[activeTab].text = "";
      tabs[activeTab].media = [];
      loadedTabs[activeTab] = true;
      renderAttachments();
      try {{
        const res = await fetch(API, {{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify({{action:"clear", tab:activeTab, active_tab:activeTab, source:"human", allow_empty:true}})}});
        const data = await res.json();
        if (data && data.state) mergeState(data.state, true);
      }} catch (err) {{}}
      dirty = false;
      updateStatus("已清空");
    }};
    fileInput.onchange = () => {{ for (const file of fileInput.files || []) readFile(file); fileInput.value = ""; }};
    document.getElementById("link").onclick = openLinkPrompt;
    document.getElementById("mergePending").onclick = async () => {{
      const pending = pendingRemote[activeTab];
      if (!pending) return;
      const current = editor.innerHTML.trim();
      const incoming = (pending.html || "").trim();
      editor.innerHTML = [current, incoming].filter(Boolean).join("<hr>");
      loadedTabs[activeTab] = true;
      tabs[activeTab].media = [...(tabs[activeTab].media || []), ...(pending.media || [])].slice(0, 8);
      pendingRemote[activeTab] = null;
      focused = false;
      dirty = true;
      renderAttachments();
      renderPending();
      await saveNow();
    }};
    document.getElementById("replacePending").onclick = async () => {{
      const pending = pendingRemote[activeTab];
      if (!pending) return;
      editor.innerHTML = pending.html || "";
      loadedTabs[activeTab] = true;
      tabs[activeTab].media = pending.media || [];
      tabs[activeTab].version = pending.version || tabs[activeTab].version;
      pendingRemote[activeTab] = null;
      focused = false;
      dirty = true;
      renderAttachments();
      renderPending();
      await saveNow();
    }};
    applyTab(true);
    poll();
    loadLiveSessions(true);
    setInterval(poll, 700);
    setInterval(() => loadLiveSessions(false), 10000);
    setInterval(loadLiveSession, 2000);
    setInterval(flushContextPendingIfReady, 900);
  </script>
</body>
</html>"""


def render_tool_dashboard_html(rows: list[dict[str, Any]], fillers: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        tool = str(row.get("tool_name") or "")
        speech_enabled = bool(row.get("speech_enabled", 1))
        table_rows.append(
            f"<tr data-tool='{html.escape(tool)}'>"
            f"<td class='tool'>{html.escape(tool)}</td>"
            f"<td class='toggle'><input type='checkbox' name='speech_enabled' value='1' {'checked' if speech_enabled else ''}></td>"
            f"<td><input name='start_phrase' value='{html.escape(str(row.get('start_phrase') or ''))}'></td>"
            f"<td><input name='success_phrase' value='{html.escape(str(row.get('success_phrase') or ''))}'></td>"
            f"<td><input name='failure_phrase' value='{html.escape(str(row.get('failure_phrase') or ''))}'></td>"
            "<td class='row-status'>自动保存</td>"
            "</tr>"
        )
    filler_sections = []
    for stage in ["opening", "working", "transition", "blocked"]:
        stage_rows = []
        for row in [r for r in fillers if str(r.get("stage") or "opening") == stage]:
            fid = int(row.get("id") or 0)
            ok = bool(row.get("ok"))
            stage_rows.append(
                f"<tr data-filler-id='{fid}'>"
                f"<td class='tool'>{fid}</td>"
                f"<td><input name='phrase' value='{html.escape(str(row.get('phrase') or ''))}'></td>"
                f"<td><select name='tone'><option value='soft' {'selected' if row.get('tone') == 'soft' else ''}>soft</option><option value='active' {'selected' if row.get('tone') == 'active' else ''}>active</option></select></td>"
                f"<td><select name='stage'>{''.join(f'<option value={json.dumps(s)} {'selected' if stage == s else ''}>{s}</option>' for s in ['opening','working','transition','blocked'])}</select></td>"
                f"<td><input name='instructions' value='{html.escape(str(row.get('instructions') or ''))}'></td>"
                f"<td class='cache'>{'ready' if ok else 'empty'}</td>"
                f"<td><span class='row-status'>自动保存</span> <button type='button' onclick='warmFiller({fid})'>预热</button> <button type='button' class='danger' onclick='deleteFiller({fid})'>删除</button></td>"
                "</tr>"
            )
        filler_sections.append(
            f"<section><h2>{stage}</h2><table><thead><tr><th>ID</th><th>短语</th><th>口气</th><th>阶段</th><th>指令</th><th>缓存</th><th></th></tr></thead><tbody>{''.join(stage_rows)}</tbody></table></section>"
        )
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Jen Tool Speech</title>
  <style>
    :root {{ color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f6f6f3; color: #191916; }}
    header {{ position: sticky; top: 0; padding: 18px 22px; background: rgba(246,246,243,.94); border-bottom: 1px solid #d8d6cf; backdrop-filter: blur(10px); }}
    h1 {{ margin: 0; font-size: 18px; font-weight: 650; }}
    main {{ padding: 18px 22px 40px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d8d6cf; }}
    th, td {{ border-bottom: 1px solid #e5e2da; padding: 8px; text-align: left; vertical-align: middle; }}
    th {{ font-size: 12px; color: #68665f; background: #fbfaf6; }}
    .tool {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; white-space: nowrap; }}
    input, select {{ box-sizing: border-box; width: 100%; min-width: 84px; padding: 7px 8px; border: 1px solid #c8c5bb; border-radius: 6px; background: #fff; color: #191916; font-size: 14px; }}
    input[type="checkbox"] {{ width: 18px; min-width: 18px; height: 18px; }}
    .toggle {{ text-align: center; width: 52px; }}
    button {{ padding: 7px 10px; border: 1px solid #222; border-radius: 6px; background: #222; color: #fff; cursor: pointer; }}
    button.danger {{ background: #8d2b22; border-color: #8d2b22; }}
    .row-status {{ color: #68665f; font-size: 12px; white-space: nowrap; }}
    .row-status.saving {{ color: #947100; }}
    .row-status.saved {{ color: #2f6f46; }}
    .row-status.error {{ color: #8d2b22; }}
    .status {{ margin-left: 12px; font-size: 13px; color: #68665f; }}
    h2 {{ margin: 28px 0 10px; font-size: 15px; }}
    .cache {{ font-size: 12px; color: #68665f; white-space: nowrap; }}
    .pipeline {{ margin: 0 0 22px; background: #fff; border: 1px solid #d8d6cf; padding: 12px; }}
    .pipeline-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; }}
    .pipeline h2 {{ margin: 0; }}
    .turn-list {{ display: grid; gap: 8px; margin-bottom: 10px; }}
    .turn-card {{ border: 1px solid #ebe7de; border-radius: 6px; background: #fbfaf6; padding: 8px; }}
    .turn-title {{ display: flex; justify-content: space-between; gap: 10px; color: #68665f; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; margin-bottom: 6px; }}
    .timing-row {{ display: grid; grid-template-columns: 110px 1fr 52px; gap: 8px; align-items: center; font-size: 12px; margin: 4px 0; }}
    .timing-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .timing-bar {{ height: 8px; background: #ece8dd; border-radius: 999px; overflow: hidden; }}
    .timing-fill {{ height: 100%; background: #2d6cdf; border-radius: inherit; }}
    .timing-fill.warn {{ background: #b56b16; }}
    .timing-fill.bad {{ background: #a83b32; }}
    .timing-sec {{ color: #68665f; text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .pipeline-list {{ display: grid; gap: 6px; max-height: 260px; overflow: auto; font-size: 12px; }}
    .pipe-item {{ display: grid; grid-template-columns: 84px 110px 1fr; gap: 8px; padding: 6px 8px; border: 1px solid #ebe7de; border-radius: 6px; background: #fbfaf6; }}
    .pipe-kind, .pipe-time {{ color: #68665f; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .pipe-content {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: #181816; color: #eee9df; }}
      header {{ background: rgba(24,24,22,.94); border-color: #3a3833; }}
      table {{ background: #20201d; border-color: #3a3833; }}
      .pipeline {{ background: #20201d; border-color: #3a3833; }}
      .turn-card {{ background: #25251f; border-color: #34322d; }}
      .timing-bar {{ background: #34322d; }}
      .pipe-item {{ background: #25251f; border-color: #34322d; }}
      th {{ background: #25251f; color: #aaa59a; }}
      th, td {{ border-color: #34322d; }}
      input, select {{ background: #181816; color: #eee9df; border-color: #4b4942; }}
    }}
  </style>
</head>
<body>
  <header><h1>Tool Speech Dashboard <span id="status" class="status"></span></h1></header>
  <main>
    <section class="pipeline">
      <div class="pipeline-head">
        <h2>最近流水线</h2>
        <button type="button" onclick="loadPipeline()">刷新</button>
      </div>
      <div id="turn-list" class="turn-list"></div>
      <div id="pipeline-list" class="pipeline-list"></div>
    </section>
    <table>
      <thead><tr><th>tool</th><th>语音</th><th>开始</th><th>成功</th><th>失败</th><th></th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
    <section>
      <h2>新增 filler</h2>
      <table><tbody><tr id="new-filler">
        <td><input name="phrase" placeholder="短语，例如：我想一下。"></td>
        <td><select name="tone"><option value="soft">soft</option><option value="active">active</option></select></td>
        <td><select name="stage"><option value="opening">opening</option><option value="working">working</option><option value="transition">transition</option><option value="blocked">blocked</option></select></td>
        <td><input name="instructions" placeholder="可选 TTS 指令"></td>
        <td><button type="button" onclick="addFiller()">新增</button></td>
      </tr></tbody></table>
    </section>
    {''.join(filler_sections)}
  </main>
  <script>
    const saveTimers = new Map();
    function setRowStatus(row, text, cls='') {{
      const el = row?.querySelector('.row-status');
      if (!el) return;
      el.className = 'row-status' + (cls ? ' ' + cls : '');
      el.textContent = text;
    }}
    function debounceSave(key, fn) {{
      clearTimeout(saveTimers.get(key));
      saveTimers.set(key, setTimeout(fn, 600));
    }}
    function rowData(row) {{
      return Object.fromEntries([...row.querySelectorAll('input, select')].map(i => [i.name, i.type === 'checkbox' ? i.checked : i.value]));
    }}
    async function saveTool(tool) {{
      const row = document.querySelector('tr[data-tool=' + CSS.escape(tool) + ']');
      if (!row) return;
      const data = rowData(row);
      setRowStatus(row, '保存中', 'saving');
      const res = await fetch('/api/tools/' + encodeURIComponent(tool), {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(data)
      }});
      const status = document.getElementById('status');
      if (!res.ok) {{
        status.textContent = '保存失败';
        setRowStatus(row, '失败', 'error');
        return;
      }}
      status.textContent = tool + ' 已保存';
      setRowStatus(row, '已保存', 'saved');
    }}
    async function saveFiller(id) {{
      const row = document.querySelector('tr[data-filler-id="' + id + '"]');
      if (!row) return;
      const data = rowData(row);
      setRowStatus(row, '保存中', 'saving');
      const res = await fetch('/api/fillers/' + id, {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(data)}});
      document.getElementById('status').textContent = res.ok ? ('filler ' + id + ' 已保存') : '保存失败';
      setRowStatus(row, res.ok ? '已保存' : '失败', res.ok ? 'saved' : 'error');
    }}
    async function warmFiller(id) {{
      document.getElementById('status').textContent = 'filler ' + id + ' 预热中';
      const res = await fetch('/api/fillers/' + id + '/warm', {{method: 'POST'}});
      const data = await res.json();
      document.getElementById('status').textContent = data.ok ? ('filler ' + id + ' 已预热') : ('预热失败: ' + (data.error || ''));
      if (data.ok) setTimeout(() => location.reload(), 500);
    }}
    async function addFiller() {{
      const row = document.getElementById('new-filler');
      const data = Object.fromEntries([...row.querySelectorAll('input, select')].map(i => [i.name, i.value]));
      const res = await fetch('/api/fillers', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(data)}});
      const out = await res.json();
      document.getElementById('status').textContent = out.ok ? ('filler ' + out.filler_id + ' 已新增') : ('新增失败: ' + (out.error || ''));
      if (out.ok) setTimeout(() => location.reload(), 400);
    }}
    async function deleteFiller(id) {{
      if (!confirm('删除 filler ' + id + ' ?')) return;
      const res = await fetch('/api/fillers/' + id, {{method: 'DELETE'}});
      const out = await res.json();
      document.getElementById('status').textContent = out.ok ? ('filler ' + id + ' 已删除') : ('删除失败: ' + (out.error || ''));
      if (out.ok) setTimeout(() => location.reload(), 400);
    }}
    function shortTime(ts) {{
      if (!ts) return '';
      return new Date(ts * 1000).toLocaleTimeString([], {{hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'}});
    }}
    function esc(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    }}
    async function loadPipeline() {{
      const box = document.getElementById('pipeline-list');
      const turnBox = document.getElementById('turn-list');
      if (!box) return;
      try {{
        const res = await fetch('/api/pipeline?limit=120');
        const data = await res.json();
        const turns = (data.turns || []).slice(0, 5);
        if (turnBox) {{
          turnBox.innerHTML = turns.map(turn => {{
            const timings = turn.timings || [];
            const maxDur = Math.max(1, ...timings.map(item => Number((item.metadata || {{}}).duration_seconds || 0)));
            const total = Math.max(0, Number(turn.last_at || 0) - Number(turn.started_at || 0));
            const timingHtml = timings.map(item => {{
              const meta = item.metadata || {{}};
              const dur = Number(meta.duration_seconds || 0);
              const pct = Math.max(2, Math.min(100, (dur / maxDur) * 100));
              const cls = dur >= 8 ? 'bad' : (dur >= 3 ? 'warn' : '');
              const name = meta.label || meta.stage || item.content || 'timing';
              return `<div class="timing-row" title="${{esc(JSON.stringify(meta).slice(0, 900))}}">
                <div class="timing-label">${{esc(name)}} <span class="pipe-kind">${{esc(meta.status || '')}}</span></div>
                <div class="timing-bar"><div class="timing-fill ${{cls}}" style="width:${{pct}}%"></div></div>
                <div class="timing-sec">${{dur ? dur.toFixed(2) + 's' : '...'}}</div>
              </div>`;
            }}).join('');
            return `<div class="turn-card">
              <div class="turn-title"><span>${{esc(turn.turn_id || '')}}</span><span>total ${{total.toFixed(2)}}s</span></div>
              ${{timingHtml || '<div class="timing-row"><div>no timing</div><div></div><div></div></div>'}}
            </div>`;
          }}).join('');
        }}
        const items = (data.items || []).slice(-40).reverse();
        box.innerHTML = items.map(item => {{
          const meta = item.metadata || {{}};
          const label = meta.tool_name || item.kind || '';
          const content = item.content || meta.summary || meta.error || '';
          const turn = item.turn_id ? item.turn_id.slice(-8) : '';
          return `<div class="pipe-item" title="${{esc(JSON.stringify(meta).slice(0, 900))}}">
            <div class="pipe-time">${{shortTime(item.created_at)}} ${{esc(turn)}}</div>
            <div class="pipe-kind">${{esc(label)}}</div>
            <div class="pipe-content">${{esc(content)}}</div>
          </div>`;
        }}).join('') || '<div class="pipe-item"><div></div><div>empty</div><div>还没有流水线事件</div></div>';
      }} catch (err) {{
        box.innerHTML = '<div class="pipe-item"><div></div><div>error</div><div>' + esc(err) + '</div></div>';
      }}
    }}
    document.querySelectorAll('tr[data-tool] input').forEach(el => {{
      const eventName = el.type === 'checkbox' ? 'change' : 'input';
      el.addEventListener(eventName, () => {{
        const row = el.closest('tr[data-tool]');
        const tool = row.dataset.tool;
        setRowStatus(row, '未保存', 'saving');
        debounceSave('tool:' + tool, () => saveTool(tool));
      }});
    }});
    document.querySelectorAll('tr[data-filler-id] input, tr[data-filler-id] select').forEach(el => {{
      const eventName = el.tagName === 'SELECT' ? 'change' : 'input';
      el.addEventListener(eventName, () => {{
        const row = el.closest('tr[data-filler-id]');
        const id = row.dataset.fillerId;
        setRowStatus(row, '未保存', 'saving');
        debounceSave('filler:' + id, () => saveFiller(id));
      }});
    }});
    loadPipeline();
    setInterval(loadPipeline, 3000);
  </script>
</body>
</html>"""


def render_domain_probe_debug_html() -> str:
    return """<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Jen Domain Probe</title>
  <style>
    :root { color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f6f3; color: #191916; }
    header { position: sticky; top: 0; z-index: 2; padding: 16px 20px; background: rgba(246,246,243,.94); border-bottom: 1px solid #d8d6cf; backdrop-filter: blur(10px); }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    main { padding: 18px 20px 42px; display: grid; gap: 16px; }
    .panel { background: #fff; border: 1px solid #d8d6cf; border-radius: 8px; padding: 14px; }
    .input-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: start; }
    textarea { width: 100%; min-height: 92px; resize: vertical; box-sizing: border-box; padding: 12px; border: 1px solid #c8c5bb; border-radius: 8px; background: #fff; color: #191916; font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    button { min-height: 42px; padding: 0 16px; border: 1px solid #222; border-radius: 8px; background: #222; color: #fff; cursor: pointer; font-weight: 600; }
    .hint { margin-top: 8px; color: #6b6860; font-size: 12px; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
    .metric { border: 1px solid #ebe7de; border-radius: 7px; padding: 10px; background: #fbfaf6; }
    .metric-label { color: #6b6860; font-size: 12px; margin-bottom: 4px; }
    .metric-value { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 14px; overflow-wrap: anywhere; }
    .domain-list, .tool-list { display: grid; gap: 10px; }
    .domain-card, .tool-card { border: 1px solid #ebe7de; border-radius: 7px; padding: 10px; background: #fbfaf6; }
    .domain-head, .tool-head { display: flex; gap: 10px; align-items: baseline; justify-content: space-between; margin-bottom: 8px; }
    .probe-count-toggle { min-height: 28px; padding: 0 10px; border-color: #d7d2c6; border-radius: 999px; background: #fff; color: #514f49; font-size: 12px; }
    .probe-count-toggle.active { border-color: #222; color: #191916; }
    .lane-log-body { display: none; }
    .lane-log-body.show { display: grid; }
    .name { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 700; }
    .muted { color: #6b6860; font-size: 12px; }
    .pill { display: inline-flex; align-items: center; border: 1px solid #d7d2c6; border-radius: 999px; padding: 2px 7px; font-size: 12px; color: #514f49; background: #fff; }
    pre { margin: 8px 0 0; padding: 10px; overflow: auto; border-radius: 7px; background: #1e1f22; color: #f4f1e8; font-size: 12px; line-height: 1.45; }
    .empty { color: #6b6860; padding: 8px 0; }
    .error { color: #8d2b22; }
    nav { margin-top: 6px; font-size: 12px; }
    nav a { color: inherit; margin-right: 12px; }
    @media (prefers-color-scheme: dark) {
      body { background: #181816; color: #eee9df; }
      header { background: rgba(24,24,22,.94); border-color: #3a3833; }
      .panel { background: #20201d; border-color: #3a3833; }
      textarea { background: #181816; color: #eee9df; border-color: #4b4942; }
      .domain-card, .tool-card, .metric { background: #25251f; border-color: #34322d; }
      .pill { background: #181816; border-color: #4b4942; color: #c9c2b4; }
      .probe-count-toggle { background: #181816; border-color: #4b4942; color: #c9c2b4; }
      .probe-count-toggle.active { border-color: #eee9df; color: #eee9df; }
      .muted, .hint, .metric-label, .empty { color: #aaa59a; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Domain Probe Debug</h1>
    <nav><a href="/">Tool speech</a><a href="/front-note">Front note</a><a href="/probe">Probe</a></nav>
  </header>
  <main>
    <section class="panel">
      <div class="input-row">
        <textarea id="message" placeholder="输入要 probe 的用户消息，例如：今天天气怎么样 / 关掉 inna / 在便签写下明天开会"></textarea>
        <button id="run" type="button">Probe</button>
      </div>
      <div class="hint">Enter 运行，Shift+Enter 换行。这个接口只跑本地 domain probe，不执行工具。</div>
    </section>
    <section class="panel">
      <div class="summary" id="summary"></div>
    </section>
    <section class="panel">
      <div class="domain-head">
        <h2>Lane Call Log</h2>
        <div>
          <button class="probe-count-toggle" id="lane-log-toggle" type="button">计数</button>
          <span class="muted" id="lane-log-meta">recent 300</span>
        </div>
      </div>
      <div class="summary lane-log-body" id="lane-log"></div>
    </section>
    <section class="panel">
      <h2>Tool Suggestions</h2>
      <div class="tool-list" id="tools"></div>
    </section>
    <section class="panel">
      <h2>Domains</h2>
      <div class="domain-list" id="domains"></div>
    </section>
    <section class="panel">
      <h2>Raw JSON</h2>
      <pre id="raw">{}</pre>
    </section>
  </main>
  <script>
    const message = document.getElementById('message');
    const run = document.getElementById('run');
    const summary = document.getElementById('summary');
    const tools = document.getElementById('tools');
    const domains = document.getElementById('domains');
    const raw = document.getElementById('raw');
    const laneLog = document.getElementById('lane-log');
    const laneLogToggle = document.getElementById('lane-log-toggle');
    const laneLogMeta = document.getElementById('lane-log-meta');
    let laneLogVisible = false;
    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    function pretty(value) {
      return esc(JSON.stringify(value ?? {}, null, 2));
    }
    function render(data) {
      const probe = data.probe || {};
      const list = data.tool_suggestions || [];
      summary.innerHTML = `
        <div class="metric"><div class="metric-label">message</div><div class="metric-value">${esc(data.message || '')}</div></div>
        <div class="metric"><div class="metric-label">duration</div><div class="metric-value">${esc(probe.duration_ms ?? '')} ms</div></div>
        <div class="metric"><div class="metric-label">domains</div><div class="metric-value">${esc((probe.domains || []).length)}</div></div>
        <div class="metric"><div class="metric-label">tool suggestions</div><div class="metric-value">${esc(list.length)}</div></div>
      `;
      tools.innerHTML = list.length ? list.map(item => `
        <div class="tool-card">
          <div class="tool-head">
            <div><span class="name">${esc(item.domain || '')}</span> <span class="pill">${esc(item.intent || '')}</span></div>
            <div class="muted">${esc(item.domain || '')} · ${esc(item.confidence ?? '')}</div>
          </div>
          <div class="muted">${esc(item.desc || '')}</div>
          <pre>${esc(item.tool_call || '')}</pre>
          ${item.answer ? `<pre>${pretty(item.answer)}</pre>` : '<div class="empty">answer pending or unsupported</div>'}
        </div>
      `).join('') : '<div class="empty">没有 tool suggestion</div>';
      domains.innerHTML = (probe.domains || []).length ? (probe.domains || []).map(item => `
        <div class="domain-card">
          <div class="domain-head">
            <div><span class="name">${esc(item.domain || '')}</span> <span class="pill">${esc(item.intent || '')}</span></div>
            <div class="muted">confidence ${esc(item.confidence ?? '')}</div>
          </div>
          <div>${esc(item.context || '')}</div>
          <pre>${pretty({matched_entities:item.matched_entities || [], suggested_actions:item.suggested_actions || []})}</pre>
        </div>
      `).join('') : '<div class="empty">没有匹配 domain</div>';
      raw.textContent = JSON.stringify(data, null, 2);
    }
    function renderCountGroup(title, rows) {
      const items = (rows || []).slice(0, 8);
      return `<div class="metric"><div class="metric-label">${esc(title)}</div><div class="metric-value">${
        items.length ? items.map(item => `${esc(item.name)}:${esc(item.count)}`).join('<br>') : '<span class="muted">empty</span>'
      }</div></div>`;
    }
    async function loadLaneLog() {
      if (!laneLog) return;
      try {
        const res = await fetch('/api/lane-log?limit=300', {cache:'no-store'});
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);
        laneLog.innerHTML = [
          renderCountGroup('calls', data.calls),
          renderCountGroup('lanes', data.lanes),
          renderCountGroup('domains', data.domains),
          renderCountGroup('tools', data.tools),
          renderCountGroup('timings', data.timings),
        ].join('');
        if (laneLogMeta) laneLogMeta.textContent = `session ${data.session_id || ''} · recent ${data.limit || 300}`;
      } catch (err) {
        laneLog.innerHTML = `<div class="metric error">${esc(err)}</div>`;
      }
    }
    function syncLaneLogToggle() {
      laneLog.classList.toggle('show', laneLogVisible);
      laneLogToggle.classList.toggle('active', laneLogVisible);
      laneLogToggle.textContent = laneLogVisible ? '隐藏计数' : '计数';
    }
    async function runProbe() {
      const text = message.value.trim();
      run.disabled = true;
      run.textContent = 'Running';
      try {
        const res = await fetch('/api/domain-probe', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:text})});
        const data = await res.json();
        if (!res.ok || data.ok === false) throw new Error(data.error || res.statusText);
        render(data);
      } catch (err) {
        summary.innerHTML = `<div class="metric error">${esc(err)}</div>`;
      } finally {
        run.disabled = false;
        run.textContent = 'Probe';
      }
    }
    run.addEventListener('click', runProbe);
    laneLogToggle.addEventListener('click', () => {
      laneLogVisible = !laneLogVisible;
      syncLaneLogToggle();
    });
    syncLaneLogToggle();
    message.addEventListener('keydown', event => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        runProbe();
      }
    });
    const params = new URLSearchParams(location.search);
    const initial = params.get('q') || params.get('message') || '';
    if (initial) {
      message.value = initial;
      runProbe();
    } else {
      render({ok:true, message:'', probe:{domains:[], duration_ms:0}, tool_suggestions:[]});
    }
    loadLaneLog();
    setInterval(loadLaneLog, 5000);
  </script>
</body>
</html>"""
