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

let indexedDocCount = 0;

function setChatStatus(message, state = "muted") {
  chatIndexStatus.textContent = message;
  chatIndexStatus.dataset.state = state;
}

function appendMessage(role, text) {
  const row = document.createElement("div");
  row.className = role === "user" ? "msg user" : "msg";

  const avatar = document.createElement("div");
  avatar.className = role === "user" ? "avatar usr" : "avatar ai";
  avatar.textContent = role === "user" ? "T" : "AI";

  const bubble = document.createElement("div");
  bubble.className = role === "user" ? "bubble usr" : "bubble ai";
  bubble.textContent = text;

  row.append(avatar, bubble);
  chatHistory.append(row);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return bubble;
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

      // Update badge and close dropdown after successful index
      indexedDocCount += data.files.length;
      updateKbBadge(indexedDocCount);
      setTimeout(closeKbDropdown, 800);

      appendMessage("ai", "Documents indexed. Ask me anything and I will answer from the retrieved context.");
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
    const pendingBubble = appendMessage("ai", "Searching documents...");

    try {
      const response = await fetch("/api/chat/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await response.json();

      if (!response.ok) throw new Error(data.error || "Chat request failed.");
      pendingBubble.textContent = data.answer || "No answer returned.";
    } catch (error) {
      pendingBubble.textContent = error.message;
    } finally {
      chatInput.disabled = false;
      chatInput.focus();
    }
  });
}

// ── PII Extractor ─────────────────────────────────────────────────────────────
const piiSystemPrompt = document.querySelector("#pii-system-prompt");
const piiInputText    = document.querySelector("#pii-input-text");
const piiExtractBtn   = document.querySelector("#pii-extract-btn");
const piiOutput       = document.querySelector("#pii-output");
const piiDownloadBtn  = document.querySelector("#pii-download-btn");
const piiCopyBtn      = document.querySelector("#pii-copy-btn");

let lastPiiData = null;

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
  if (typeof value === "object") return JSON.stringify(value) ?? String(value);
  return String(value);
}

function renderPiiTable(data) {
  if (!piiOutput) return;
  const entities = normalizePiiEntities(data);
  if (!entities.length) {
    piiOutput.innerHTML = '<div class="table-empty">No PII entities found.</div>';
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

    if (!text)         { piiOutput.textContent = "❌ Error: Input text is required."; return; }
    if (!systemPrompt) { piiOutput.textContent = "❌ Error: System prompt is required."; return; }

    piiExtractBtn.disabled = true;
    piiOutput.textContent = "⏳ Extracting PII...";

    try {
      const response = await fetch("/api/pii/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, system_prompt: systemPrompt }),
      });
      const data = await response.json();

      if (!response.ok) throw new Error(data.error || "PII extraction failed.");

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
    if (!lastPiiData) { alert("Extract PII first"); return; }
    navigator.clipboard.writeText(JSON.stringify(lastPiiData, null, 2)).then(() => {
      piiCopyBtn.textContent = "✓ Copied!";
      setTimeout(() => { piiCopyBtn.textContent = "Copy JSON"; }, 2000);
    });
  });
}

if (piiDownloadBtn) {
  piiDownloadBtn.addEventListener("click", () => {
    if (!lastPiiData) { alert("Extract PII first"); return; }
    const blob = new Blob([JSON.stringify(lastPiiData, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url;
    a.download = "pii_extraction.json";
    a.click();
    URL.revokeObjectURL(url);
  });
}
