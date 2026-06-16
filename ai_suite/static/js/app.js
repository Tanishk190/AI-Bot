// ── Consent banner ───────────────────────────────────────────────────────────
const consentBanner = document.querySelector("#consent-banner");
const consentAccept = document.querySelector("#consent-accept");
if (consentBanner && !localStorage.getItem("docuMindConsent")) {
  consentBanner.hidden = false;
}
if (consentAccept) {
  consentAccept.addEventListener("click", () => {
    localStorage.setItem("docuMindConsent", "1");
    consentBanner.hidden = true;
  });
}

// ── Panel navigation ──────────────────────────────────────────────────────────
const panels    = document.querySelectorAll(".panel");
const navItems  = document.querySelectorAll(".nav-item");
const pageTitle = document.querySelector("#page-title");
const pageBadge = document.querySelector("#page-badge");
const kbWrap    = document.querySelector("#kb-dropdown-wrap");

function showPanel(item) {
  const id = item.dataset.panel;

  panels.forEach((p) => p.classList.remove("active"));
  navItems.forEach((n) => n.classList.remove("active"));

  document.querySelector(`#panel-${id}`).classList.add("active");
  item.classList.add("active");
  pageTitle.textContent = item.dataset.title;
  pageBadge.textContent = item.dataset.badge;

  // Show KB dropdown button only on chat panel
  if (kbWrap) {
    kbWrap.classList.toggle("hidden", id !== "chat");
    if (id !== "chat") closeKbDropdown();
  }
}

navItems.forEach((item) => item.addEventListener("click", () => showPanel(item)));

// ── Mobile sidebar drawer ─────────────────────────────────────────────────────
const appShell      = document.querySelector("#app-shell");
const menuToggle    = document.querySelector("#menu-toggle");
const sidebarOverlay = document.querySelector("#sidebar-overlay");

function openSidebar() {
  appShell.classList.add("nav-open");
  if (menuToggle) menuToggle.setAttribute("aria-expanded", "true");
}
function closeSidebar() {
  appShell.classList.remove("nav-open");
  if (menuToggle) menuToggle.setAttribute("aria-expanded", "false");
}

if (menuToggle) {
  menuToggle.addEventListener("click", () => {
    appShell.classList.contains("nav-open") ? closeSidebar() : openSidebar();
  });
}
if (sidebarOverlay) sidebarOverlay.addEventListener("click", closeSidebar);
// Close the drawer after picking a tool on mobile
navItems.forEach((item) => item.addEventListener("click", closeSidebar));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSidebar(); });

// ── Desktop sidebar collapse ──────────────────────────────────────────────────
const collapseToggle = document.querySelector("#collapse-toggle");

function applyCollapsed(collapsed) {
  appShell.classList.toggle("sidebar-collapsed", collapsed);
  if (collapseToggle) {
    collapseToggle.setAttribute("aria-pressed", String(collapsed));
    collapseToggle.setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
    collapseToggle.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
  }
}

applyCollapsed(localStorage.getItem("docuMindSidebarCollapsed") === "1");

if (collapseToggle) {
  collapseToggle.addEventListener("click", () => {
    const collapsed = !appShell.classList.contains("sidebar-collapsed");
    applyCollapsed(collapsed);
    localStorage.setItem("docuMindSidebarCollapsed", collapsed ? "1" : "0");
  });
}

// ── Knowledge Base dropdown ───────────────────────────────────────────────────
const kbToggleBtn = document.querySelector("#kb-toggle-btn");
const kbDropdown  = document.querySelector("#kb-dropdown");
const kbDocBadge  = document.querySelector("#kb-doc-badge");

function openKbDropdown() {
  if (!kbToggleBtn || !kbDropdown) return;
  kbToggleBtn.setAttribute("aria-expanded", "true");
  kbDropdown.setAttribute("aria-hidden", "false");
}

function closeKbDropdown() {
  if (!kbToggleBtn || !kbDropdown) return;
  kbToggleBtn.setAttribute("aria-expanded", "false");
  kbDropdown.setAttribute("aria-hidden", "true");
}

function toggleKbDropdown() {
  const isOpen = kbToggleBtn.getAttribute("aria-expanded") === "true";
  isOpen ? closeKbDropdown() : openKbDropdown();
}

if (kbToggleBtn) {
  kbToggleBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleKbDropdown();
  });
}

// Close dropdown when clicking outside
document.addEventListener("click", (e) => {
  if (kbWrap && !kbWrap.contains(e.target)) closeKbDropdown();
});

// Close on Escape key
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeKbDropdown();
});

// Update doc count badge on the toggle button
function updateKbBadge(count) {
  if (!kbDocBadge) return;
  if (count > 0) {
    kbDocBadge.textContent = `${count} doc${count === 1 ? "" : "s"}`;
    kbDocBadge.hidden = false;
  } else {
    kbDocBadge.hidden = true;
  }
}

// ── AI Chat ───────────────────────────────────────────────────────────────────
const chatDocuments   = document.querySelector("#chat-documents");
const chatIndexButton = document.querySelector("#chat-index-btn");
const chatIndexStatus = document.querySelector("#chat-index-status");
const chatHistory     = document.querySelector("#chat-history");
const chatForm        = document.querySelector("#chat-form");
const chatInput       = document.querySelector("#chat-input");
const chatClearBtn    = document.querySelector("#chat-clear-btn");

// Snapshot of the default welcome message, restored when the chat is cleared
const chatWelcomeHTML = chatHistory ? chatHistory.innerHTML : "";

let indexedDocCount = 0;
const kbDocItems    = document.querySelector("#kb-doc-items");
const kbNoDocs      = document.querySelector("#kb-no-docs");
const kbSelectAll   = document.querySelector("#kb-select-all");
const kbSelectWrap  = document.querySelector("#kb-select-all-wrap");

function setChatStatus(message, state = "muted") {
  chatIndexStatus.textContent = message;
  chatIndexStatus.dataset.state = state;
}

// ── Document list ────────────────────────────────────────────────────────────
let kbDocData = [];

async function loadDocumentList() {
  try {
    const response = await fetch("/api/chat/documents");
    const data = await response.json();
    kbDocData = data.documents || [];
    renderDocumentList();
  } catch (e) {
    console.error("Failed to load documents:", e);
  }
}

function renderDocumentList() {
  if (!kbDocItems) return;
  kbDocItems.innerHTML = "";

  if (!kbDocData.length) {
    kbDocItems.innerHTML = '<div class="table-empty">No documents indexed yet.</div>';
    if (kbSelectWrap) kbSelectWrap.hidden = true;
    updateKbBadge(0);
    return;
  }

  if (kbSelectWrap) kbSelectWrap.hidden = false;
  updateKbBadge(kbDocData.length);

  kbDocData.forEach((doc) => {
    const item = document.createElement("div");
    item.className = "kb-doc-item";

    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.value = doc.id;
    cb.className = "kb-doc-cb";

    const name = document.createElement("span");
    name.className = "doc-filename";
    name.textContent = doc.filename;
    name.title = doc.filename;

    const chunks = document.createElement("span");
    chunks.className = "doc-chunks";
    chunks.textContent = `${doc.chunk_count} chunks`;

    const delBtn = document.createElement("button");
    delBtn.className = "kb-doc-del";
    delBtn.textContent = "×";
    delBtn.title = "Remove document";
    delBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      delBtn.disabled = true;
      try {
        const res = await fetch(`/api/chat/documents/${doc.id}`, { method: "DELETE" });
        if (res.ok) await loadDocumentList();
      } catch (err) {
        console.error("Delete failed:", err);
      }
    });

    label.append(cb, name);
    item.append(label, chunks, delBtn);
    kbDocItems.append(item);
  });
}

function getSelectedDocumentIds() {
  if (!kbDocItems) return null;
  const checkboxes = kbDocItems.querySelectorAll(".kb-doc-cb");
  if (!checkboxes.length) return null;
  const allChecked = Array.from(checkboxes).every((cb) => cb.checked);
  if (allChecked) return null; // null = search all
  const selected = Array.from(checkboxes)
    .filter((cb) => cb.checked)
    .map((cb) => parseInt(cb.value, 10));
  return selected.length ? selected : null;
}

if (kbSelectAll) {
  kbSelectAll.addEventListener("change", () => {
    const checkboxes = kbDocItems.querySelectorAll(".kb-doc-cb");
    checkboxes.forEach((cb) => { cb.checked = kbSelectAll.checked; });
  });
}

// Load documents on page load
loadDocumentList();

// ── Markdown renderer ────────────────────────────────────────────────────────
function renderMarkdown(text) {
  if (!text) return "";
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="md-code"><code>$2</code></pre>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="md-inline">$1</code>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // Italic
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  // Headers (##)
  html = html.replace(/^### (.+)$/gm, '<div class="md-h3">$1</div>');
  html = html.replace(/^## (.+)$/gm, '<div class="md-h2">$1</div>');

  // Unordered lists (items may be separated by blank lines)
  html = html.replace(/(^|\n)(- .+(?:\n+- .+)*)/g, (_, pre, block) => {
    const items = block.split("\n")
      .filter((l) => /^-\s/.test(l.trim()))
      .map((l) => `<li>${l.trim().replace(/^-\s/, "")}</li>`)
      .join("");
    return `${pre}<ul>${items}</ul>`;
  });
  // Ordered lists — group consecutive numbered items (even across blank lines)
  // into one <ol> so the browser renumbers them 1,2,3… instead of 1,1,1…
  html = html.replace(/(^|\n)(\d+\. .+(?:\n+\d+\. .+)*)/g, (_, pre, block) => {
    const items = block.split("\n")
      .filter((l) => /^\d+\.\s/.test(l.trim()))
      .map((l) => `<li>${l.trim().replace(/^\d+\.\s/, "")}</li>`)
      .join("");
    return `${pre}<ol>${items}</ol>`;
  });

  // Line breaks (but not inside lists/pre)
  html = html.replace(/\n/g, "<br>");
  // Clean up double breaks from block elements
  html = html.replace(/<br><(ul|ol|pre|div)/g, "<$1");
  html = html.replace(/<\/(ul|ol|pre|div)><br>/g, "</$1>");

  return html;
}

// Citation pill label, e.g. "audit.pdf · p.3 · #9"
function sourceLabel(s) {
  const page = s.page ? ` · p.${s.page}` : "";
  return `${s.source}${page} · #${s.chunk}`;
}

function appendMessage(role, text, sources) {
  const row = document.createElement("div");
  row.className = role === "user" ? "msg user" : "msg";

  const avatar = document.createElement("div");
  avatar.className = role === "user" ? "avatar usr" : "avatar ai";
  avatar.textContent = role === "user" ? "T" : "AI";

  const bubble = document.createElement("div");
  bubble.className = role === "user" ? "bubble usr" : "bubble ai";

  if (role === "user") {
    bubble.textContent = text;
  } else {
    bubble.innerHTML = renderMarkdown(text);
  }

  row.append(avatar, bubble);

  // Sources pill row
  if (sources && sources.length && role !== "user") {
    const srcRow = document.createElement("div");
    srcRow.className = "source-row";
    sources.forEach((s) => {
      const pill = document.createElement("span");
      pill.className = "source-pill";
      pill.textContent = sourceLabel(s);
      srcRow.append(pill);
    });
    bubble.append(srcRow);
  }

  chatHistory.append(row);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return bubble;
}

// ── Load chat history from DB ────────────────────────────────────────────────
async function loadChatHistory() {
  try {
    const response = await fetch("/api/chat/history");
    const data = await response.json();
    if (!data.messages || !data.messages.length) return;

    // Clear the default welcome message
    if (chatHistory) chatHistory.innerHTML = "";

    data.messages.forEach((msg) => {
      const role = msg.role === "assistant" ? "ai" : "user";
      appendMessage(role, msg.content, msg.sources);
    });
  } catch (e) {
    console.error("Failed to load chat history:", e);
  }
}

loadChatHistory();

// ── Clear conversation ───────────────────────────────────────────────────────
if (chatClearBtn) {
  chatClearBtn.addEventListener("click", async () => {
    if (!confirm("Clear this conversation? This cannot be undone.")) return;
    chatClearBtn.disabled = true;
    try {
      const response = await fetch("/api/chat/history", { method: "DELETE" });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || "Failed to clear chat.");
      }
      if (chatHistory) chatHistory.innerHTML = chatWelcomeHTML;
    } catch (e) {
      console.error(e);
      alert(e.message || "Failed to clear chat.");
    } finally {
      chatClearBtn.disabled = false;
    }
  });
}

if (chatDocuments) {
  chatDocuments.addEventListener("change", () => {
    const count = chatDocuments.files.length;
    if (count === 0) {
      setChatStatus("No documents selected.");
      return;
    }
    setChatStatus(`${count} document${count === 1 ? "" : "s"} selected. Ready to index.`, "ready");
  });
}

if (chatIndexButton) {
  chatIndexButton.addEventListener("click", async () => {
    if (!chatDocuments.files.length) {
      setChatStatus("Choose PDF, TXT, or DOCX files first.", "error");
      return;
    }

    const formData = new FormData();
    Array.from(chatDocuments.files).forEach((file) => formData.append("documents", file));

    chatIndexButton.disabled = true;
    setChatStatus("Indexing documents...", "ready");

    try {
      const response = await fetch("/api/chat/index", { method: "POST", body: formData });
      const data = await response.json();

      if (!response.ok) throw new Error(data.error || "Document indexing failed.");

      const filenames = data.files.map((f) => f.name).join(", ");
      setChatStatus(`Indexed ${data.total_chunks} chunks from ${filenames}.`, "success");

      // Refresh document list and close dropdown after successful index
      await loadDocumentList();
      setTimeout(closeKbDropdown, 800);

      let msg = "Documents indexed. Ask me anything and I will answer from the retrieved context.";
      if (data.warnings && data.warnings.length) {
        msg += "\n\n⚠️ " + data.warnings.join("\n⚠️ ");
      }
      appendMessage("ai", msg);
    } catch (error) {
      setChatStatus(error.message, "error");
    } finally {
      chatIndexButton.disabled = false;
    }
  });
}

if (chatForm) {
  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    appendMessage("user", message);
    chatInput.value = "";
    chatInput.disabled = true;
    const pendingBubble = appendMessage("ai", "Searching documents…");
    pendingBubble.classList.add("streaming");

    try {
      const body = { message };
      const selectedIds = getSelectedDocumentIds();
      if (selectedIds) body.document_ids = selectedIds;

      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || "Chat request failed.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";
      let sources = [];
      let started = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line
        const frames = buffer.split("\n\n");
        buffer = frames.pop();

        for (const frame of frames) {
          const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
          if (!dataLine) continue;
          const payload = JSON.parse(dataLine.slice(5).trim());

          if (payload.error) throw new Error(payload.error);

          if (payload.delta) {
            if (!started) { answer = ""; started = true; }
            answer += payload.delta;
            pendingBubble.innerHTML = renderMarkdown(answer);
            chatHistory.scrollTop = chatHistory.scrollHeight;
          } else if (payload.done) {
            sources = payload.sources || [];
          }
        }
      }

      pendingBubble.classList.remove("streaming");
      if (!answer) pendingBubble.textContent = "No answer returned.";

      if (sources.length) {
        const srcRow = document.createElement("div");
        srcRow.className = "source-row";
        sources.forEach((s) => {
          const pill = document.createElement("span");
          pill.className = "source-pill";
          pill.textContent = sourceLabel(s);
          srcRow.append(pill);
        });
        pendingBubble.append(srcRow);
      }
    } catch (error) {
      pendingBubble.classList.remove("streaming");
      pendingBubble.textContent = error.message;
    } finally {
      chatInput.disabled = false;
      chatInput.focus();
    }
  });
}

// ── Financial Extractor ───────────────────────────────────────────────────────
const piiSystemPrompt = document.querySelector("#pii-system-prompt");
const piiInputText    = document.querySelector("#pii-input-text");
const piiExtractBtn   = document.querySelector("#pii-extract-btn");
const piiOutput       = document.querySelector("#pii-output");
const piiDownloadBtn  = document.querySelector("#pii-download-btn");
const piiCsvBtn       = document.querySelector("#pii-csv-btn");
const piiCopyBtn      = document.querySelector("#pii-copy-btn");

let lastPiiData = null;

// Stringify a scalar or array-of-scalars to a single readable cell value
function valueToText(v) {
  if (v === null || v === undefined || v === "") return "";
  if (Array.isArray(v)) return v.map(valueToText).filter(Boolean).join("; ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

// A flat field map = object whose values are all scalars or arrays of scalars
function isFlatFieldMap(data) {
  return (
    data && typeof data === "object" && !Array.isArray(data) &&
    Object.values(data).every((v) => v === null || typeof v !== "object" || Array.isArray(v))
  );
}

function renderFieldValueTable(data) {
  const table = document.createElement("table");
  table.className = "pii-table";

  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  ["Field", "Value"].forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    hr.append(th);
  });
  thead.append(hr);

  const tbody = document.createElement("tbody");
  Object.entries(data).forEach(([key, value]) => {
    const tr = document.createElement("tr");
    const tdK = document.createElement("td");
    tdK.textContent = key;
    const tdV = document.createElement("td");
    tdV.textContent = valueToText(value) || "—";
    tr.append(tdK, tdV);
    tbody.append(tr);
  });

  table.append(thead, tbody);
  piiOutput.innerHTML = "";
  piiOutput.append(table);
}

function normalizePiiEntities(data) {
  if (!data) return [];
  if (Array.isArray(data)) return data;
  if (typeof data === "object") {
    if (Array.isArray(data.entities)) return data.entities;
    const entries = Object.entries(data);
    const objectValues = entries.filter(([, v]) => v && typeof v === "object" && !Array.isArray(v));
    const entityLikeKeys = entries.filter(([k]) => /\d/.test(k) || /person|entity/i.test(k));
    if (objectValues.length === entries.length && entityLikeKeys.length >= Math.max(1, Math.ceil(entries.length / 2))) {
      return entries.map(([key, value]) => ({ entity: key, ...value }));
    }
    const arrayFields = entries.filter(([, v]) => Array.isArray(v));
    if (arrayFields.length && arrayFields.length === entries.length) {
      const lengths = arrayFields.map(([, v]) => v.length);
      const rowCount = Math.max(...lengths);
      if (lengths.every((l) => l === rowCount)) {
        return Array.from({ length: rowCount }, (_, i) => {
          const row = {};
          arrayFields.forEach(([k, vals]) => { row[k] = vals[i]; });
          return row;
        });
      }
    }
    const arrayValues = arrayFields.map(([, v]) => v);
    if (arrayValues.length === 1) return arrayValues[0];
    return [data];
  }
  return [data];
}

function formatPiiCell(value) {
  if (value === null || value === undefined || value === "") return "—";
  return valueToText(value) || "—";
}

function renderPiiTable(data) {
  if (!piiOutput) return;

  // Flat schemas (the default financial extractor) render as a Field / Value table
  if (isFlatFieldMap(data)) {
    renderFieldValueTable(data);
    return;
  }

  const entities = normalizePiiEntities(data);
  if (!entities.length) {
    piiOutput.innerHTML = '<div class="table-empty">No entities found.</div>';
    return;
  }

  const normalized = entities.map((entry) =>
    entry && typeof entry === "object" && !Array.isArray(entry) ? entry : { value: entry }
  );

  const columns = [];
  normalized.forEach((entry) => {
    Object.keys(entry).forEach((k) => { if (!columns.includes(k)) columns.push(k); });
  });
  if (!columns.length) columns.push("value");

  const table = document.createElement("table");
  table.className = "pii-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach((col) => {
    const th = document.createElement("th");
    th.textContent = col;
    headerRow.append(th);
  });
  thead.append(headerRow);

  const tbody = document.createElement("tbody");
  normalized.forEach((entry) => {
    const row = document.createElement("tr");
    columns.forEach((col) => {
      const td = document.createElement("td");
      td.textContent = formatPiiCell(entry[col]);
      row.append(td);
    });
    tbody.append(row);
  });

  table.append(thead, tbody);
  piiOutput.innerHTML = "";
  piiOutput.append(table);
}

if (piiExtractBtn) {
  piiExtractBtn.addEventListener("click", async () => {
    const text = piiInputText.value.trim();
    const systemPrompt = piiSystemPrompt.value.trim();

    if (!text)         { piiOutput.textContent = "❌ Error: Document text is required."; return; }
    if (!systemPrompt) { piiOutput.textContent = "❌ Error: Extraction instructions are required."; return; }

    piiExtractBtn.disabled = true;
    piiOutput.textContent = "⏳ Extracting…";

    try {
      const response = await fetch("/api/pii/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, system_prompt: systemPrompt }),
      });
      const data = await response.json();

      if (!response.ok) throw new Error(data.error || "Extraction failed.");

      let piiData = data.data;
      if (typeof piiData === "string") piiData = JSON.parse(piiData);
      lastPiiData = piiData;
      renderPiiTable(piiData);
    } catch (error) {
      piiOutput.textContent = `❌ Error: ${error.message}`;
    } finally {
      piiExtractBtn.disabled = false;
    }
  });
}

if (piiCopyBtn) {
  piiCopyBtn.addEventListener("click", () => {
    if (!lastPiiData) { alert("Extract first"); return; }
    navigator.clipboard.writeText(JSON.stringify(lastPiiData, null, 2)).then(() => {
      piiCopyBtn.textContent = "✓ Copied!";
      setTimeout(() => { piiCopyBtn.textContent = "Copy JSON"; }, 2000);
    });
  });
}

function downloadBlob(content, filename, type) {
  const blob = new Blob([content], { type });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

if (piiDownloadBtn) {
  piiDownloadBtn.addEventListener("click", () => {
    if (!lastPiiData) { alert("Extract first"); return; }
    downloadBlob(JSON.stringify(lastPiiData, null, 2), "extraction.json", "application/json");
  });
}

// Quote a CSV field per RFC 4180
function csvCell(value) {
  const s = valueToText(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function buildCsv(data) {
  // Flat field map → two-column Field,Value sheet
  if (isFlatFieldMap(data)) {
    const rows = [["Field", "Value"]];
    Object.entries(data).forEach(([k, v]) => rows.push([k, valueToText(v)]));
    return rows.map((r) => r.map(csvCell).join(",")).join("\r\n");
  }
  // Otherwise: tabular entity list with a column per key
  const entities = normalizePiiEntities(data).map((e) =>
    e && typeof e === "object" && !Array.isArray(e) ? e : { value: e }
  );
  const columns = [];
  entities.forEach((e) => Object.keys(e).forEach((k) => { if (!columns.includes(k)) columns.push(k); }));
  if (!columns.length) columns.push("value");
  const rows = [columns];
  entities.forEach((e) => rows.push(columns.map((c) => valueToText(e[c]))));
  return rows.map((r) => r.map(csvCell).join(",")).join("\r\n");
}

if (piiCsvBtn) {
  piiCsvBtn.addEventListener("click", () => {
    if (!lastPiiData) { alert("Extract first"); return; }
    // Prepend a UTF-8 BOM so Excel reads ₹ and other symbols correctly
    downloadBlob("﻿" + buildCsv(lastPiiData), "extraction.csv", "text/csv;charset=utf-8");
  });
}

async function parseJsonResponse(response) {
  const body = await response.text();
  if (!body) return {};
  try {
    return JSON.parse(body);
  } catch (error) {
    const message = response.ok ? "Unexpected response from server." : `Request failed (${response.status}).`;
    throw new Error(message);
  }
}

// ── OCR ───────────────────────────────────────────────────────────────────────
const ocrImages = document.querySelector("#ocr-images");
const ocrStatus = document.querySelector("#ocr-status");
const ocrRunBtn = document.querySelector("#ocr-run-btn");
const ocrResult = document.querySelector("#ocr-result");
const ocrCopyBtn = document.querySelector("#ocr-copy-btn");
const ocrDownloadBtn = document.querySelector("#ocr-download-btn");

let lastOcrText = "";

if (ocrImages) {
  ocrImages.addEventListener("change", () => {
    const count = ocrImages.files.length;
    if (!ocrStatus) return;
    if (count === 0) {
      ocrStatus.textContent = "No images selected.";
      ocrStatus.dataset.state = "";
      return;
    }
    ocrStatus.textContent = `${count} image${count === 1 ? "" : "s"} selected.`;
    ocrStatus.dataset.state = "ready";
  });
}

if (ocrRunBtn) {
  ocrRunBtn.addEventListener("click", async () => {
    if (!ocrImages || !ocrImages.files.length) {
      if (ocrStatus) {
        ocrStatus.textContent = "Choose PNG, JPG, or JPEG files first.";
        ocrStatus.dataset.state = "error";
      }
      return;
    }

    ocrRunBtn.disabled = true;
    ocrRunBtn.textContent = "Running...";
    if (ocrStatus) {
      ocrStatus.textContent = "Extracting text...";
      ocrStatus.dataset.state = "ready";
    }
    if (ocrResult) ocrResult.textContent = "⏳ Extracting...";

    try {
      const formData = new FormData();
      Array.from(ocrImages.files).forEach((file) => formData.append("images", file));

      const response = await fetch("/api/ocr/extract", {
        method: "POST",
        body: formData,
      });
      const data = await parseJsonResponse(response);

      if (!response.ok) throw new Error(data.error || `OCR failed (${response.status}).`);

      lastOcrText = data.text || "";
      if (ocrResult) ocrResult.textContent = lastOcrText || "No text found.";
      if (ocrStatus) {
        ocrStatus.textContent = "OCR complete.";
        ocrStatus.dataset.state = "success";
      }
    } catch (error) {
      if (ocrResult) ocrResult.textContent = `❌ Error: ${error.message}`;
      if (ocrStatus) {
        ocrStatus.textContent = error.message;
        ocrStatus.dataset.state = "error";
      }
    } finally {
      ocrRunBtn.disabled = false;
      ocrRunBtn.textContent = "Run OCR";
    }
  });
}

if (ocrCopyBtn) {
  ocrCopyBtn.addEventListener("click", () => {
    if (!lastOcrText) { alert("Run OCR first"); return; }
    navigator.clipboard.writeText(lastOcrText).then(() => {
      ocrCopyBtn.textContent = "✓ Copied!";
      setTimeout(() => { ocrCopyBtn.textContent = "Copy text"; }, 2000);
    });
  });
}

if (ocrDownloadBtn) {
  ocrDownloadBtn.addEventListener("click", () => {
    if (!lastOcrText) { alert("Run OCR first"); return; }
    const blob = new Blob([lastOcrText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ocr_text.txt";
    a.click();
    URL.revokeObjectURL(url);
  });
}

// ── Document Classifier ───────────────────────────────────────────────────────
const classifyDocuments = document.querySelector('input[name="classifier_documents"]');
const classifyDocList = document.querySelector("#classify-doc-list");
const classifierCriteria = document.querySelector("#criteria");
const classifyBtn = document.querySelector(".panel#panel-classify .run-btn") || 
                    document.querySelector("#panel-classify .run-btn");

if (classifyDocuments) {
  classifyDocuments.addEventListener("change", () => {
    const count = classifyDocuments.files.length;
    if (!classifyDocuments.parentElement) return;
    const uploadZone = classifyDocuments.parentElement;
    const statusText = uploadZone.querySelector("strong");
    if (statusText) {
      statusText.textContent = count === 0 
        ? "Upload documents for classification" 
        : `${count} document${count === 1 ? "" : "s"} selected`;
    }
  });
}

// Find or create classify button if it doesn't exist
let actualClassifyBtn = classifyBtn;
if (!actualClassifyBtn) {
  const panel = document.querySelector("#panel-classify");
  if (panel) {
    actualClassifyBtn = document.createElement("button");
    actualClassifyBtn.className = "run-btn";
    actualClassifyBtn.textContent = "Classify Documents";
    actualClassifyBtn.type = "button";
    
    const uploadZone = panel.querySelector(".upload-zone.compact");
    if (uploadZone) {
      uploadZone.parentElement.insertBefore(actualClassifyBtn, uploadZone.nextSibling);
    }
  }
}

if (actualClassifyBtn) {
  actualClassifyBtn.addEventListener("click", async () => {
    if (!classifyDocuments || !classifyDocuments.files.length) {
      alert("Upload at least one document first.");
      return;
    }

    if (!classifierCriteria || !classifierCriteria.value.trim()) {
      alert("Enter classification criteria first.");
      return;
    }

    actualClassifyBtn.disabled = true;
    actualClassifyBtn.textContent = "Classifying...";

    try {
      const formData = new FormData();
      Array.from(classifyDocuments.files).forEach((file) => formData.append("documents", file));
      formData.append("criteria", classifierCriteria.value.trim());

      const response = await fetch("/api/classify/documents", {
        method: "POST",
        body: formData,
      });
      const data = await parseJsonResponse(response);

      if (!response.ok) throw new Error(data.error || "Classification failed.");

      // Update doc list with results
      if (classifyDocList && data.results) {
        classifyDocList.innerHTML = "";
        data.results.forEach((result) => {
          const row = document.createElement("div");
          row.className = "doc-row";

          const fileIcon = document.createElement("span");
          fileIcon.className = "file-icon";
          fileIcon.textContent = "FILE";

          const docName = document.createElement("span");
          docName.className = "doc-name";
          docName.title = `Similarity: ${result.similarity_score.toFixed(3)}\nReasoning: ${result.reasoning}`;
          docName.textContent = result.filename;

          const tag = document.createElement("span");
          tag.className = "doc-tag";
          const classification = (result.classification || "")
            .toUpperCase()
            .replace(/[-\s]+/g, "_")
            .trim();
          if (classification === "RELEVANT") {
            tag.classList.add("tag-rel");
            tag.textContent = "Relevant";
          } else if (classification === "NOT_RELEVANT" || classification === "IRRELEVANT") {
            tag.classList.add("tag-nrel");
            tag.textContent = "Not Relevant";
          } else {
            tag.classList.add("tag-proc");
            tag.textContent = "Uncertain";
          }

          row.append(fileIcon, docName, tag);
          classifyDocList.append(row);
        });
      }

      actualClassifyBtn.textContent = `✓ Classified ${data.total_documents} document${data.total_documents === 1 ? "" : "s"}`;
      setTimeout(() => {
        actualClassifyBtn.textContent = "Classify Documents";
      }, 2000);
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      actualClassifyBtn.disabled = false;
      if (actualClassifyBtn.textContent === "Classifying...") {
        actualClassifyBtn.textContent = "Classify Documents";
      }
    }
  });
}
