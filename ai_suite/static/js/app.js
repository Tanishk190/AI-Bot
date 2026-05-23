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

// ── Sentiment Analysis ────────────────────────────────────────────────────────
const sentimentDocuments = document.querySelector("#sentiment-documents");
const sentimentFileStatus = document.querySelector("#sentiment-file-status");
const sentimentTextarea = document.querySelector("#sentiment-textarea");
const sentimentAnalyzeBtn = document.querySelector("#sentiment-analyze-btn");
const sentimentStatus = document.querySelector("#sentiment-status");
const sentimentVal = document.querySelector("#sentiment-val");
const confidenceVal = document.querySelector("#confidence-val");
const toneVal = document.querySelector("#tone-val");
const reasoningText = document.querySelector("#reasoning-text");
const positiveMeter = document.querySelector("#positive-meter");
const neutralMeter = document.querySelector("#neutral-meter");
const negativeMeter = document.querySelector("#negative-meter");
const positivePercent = document.querySelector("#positive-percent");
const neutralPercent = document.querySelector("#neutral-percent");
const negativePercent = document.querySelector("#negative-percent");

if (sentimentDocuments) {
  sentimentDocuments.addEventListener("change", () => {
    const count = sentimentDocuments.files.length;
    if (!sentimentFileStatus) return;
    if (count === 0) {
      sentimentFileStatus.textContent = "No documents selected. Click 🗁 to upload.";
      return;
    }
    if (sentimentTextarea) sentimentTextarea.value = "";
    const names = Array.from(sentimentDocuments.files).map((file) => file.name).join(", ");
    sentimentFileStatus.textContent = `Selected: ${names}`;
  });
}

if (sentimentTextarea) {
  sentimentTextarea.addEventListener("input", () => {
    const text = sentimentTextarea.value.trim();
    if (!sentimentDocuments) return;
    if (!text) {
      if (sentimentFileStatus) sentimentFileStatus.textContent = "No documents selected. Click 🗁 to upload.";
      return;
    }
    if (sentimentDocuments.files.length > 0) {
      sentimentDocuments.value = "";
      if (sentimentFileStatus) sentimentFileStatus.textContent = "No documents selected. Click 🗁 to upload.";
    }
    if (sentimentFileStatus) sentimentFileStatus.textContent = "Using text input.";
  });
}

if (sentimentAnalyzeBtn) {
  sentimentAnalyzeBtn.addEventListener("click", async () => {
    const text = sentimentTextarea ? sentimentTextarea.value.trim() : "";
    const hasDocuments = sentimentDocuments && sentimentDocuments.files.length > 0;

    if (!text && !hasDocuments) {
      if (sentimentStatus) {
        sentimentStatus.textContent = "❌ Error: Provide text or upload documents";
        sentimentStatus.dataset.state = "error";
      }
      return;
    }
    if (text && hasDocuments) {
      if (sentimentStatus) {
        sentimentStatus.textContent = "❌ Error: Provide either text or documents, not both.";
        sentimentStatus.dataset.state = "error";
      }
      return;
    }

    sentimentAnalyzeBtn.disabled = true;
    sentimentAnalyzeBtn.textContent = "Analyzing...";
    if (sentimentStatus) {
      sentimentStatus.textContent = "";
      sentimentStatus.dataset.state = "";
    }

    try {
      const formData = new FormData();
      if (text) formData.append("text", text);
      if (hasDocuments) {
        Array.from(sentimentDocuments.files).forEach((file) => formData.append("documents", file));
      }

      const response = await fetch("/api/sentiment/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await parseJsonResponse(response);

      if (!response.ok) throw new Error(data.error || `Sentiment analysis failed (${response.status}).`);

      const sentiment = data.sentiment;
      if (sentimentVal) {
        sentimentVal.textContent = sentiment;
        sentimentVal.className = "sent-val";
        sentimentVal.style.color = "";
        if (sentiment === "Positive") sentimentVal.classList.add("pos");
        if (sentiment === "Negative") sentimentVal.style.color = "var(--red)";
        if (sentiment === "Neutral") sentimentVal.style.color = "var(--gray)";
      }

      const confidence = data.confidence;
      if (confidenceVal) confidenceVal.textContent = `${confidence}%`;
      if (toneVal) toneVal.textContent = data.tone;
      if (reasoningText) reasoningText.textContent = data.reasoning;

      if (positiveMeter && neutralMeter && negativeMeter) {
        positiveMeter.style.width = sentiment === "Positive" ? `${confidence}%` : "10%";
        neutralMeter.style.width = sentiment === "Neutral" ? `${confidence}%` : "10%";
        negativeMeter.style.width = sentiment === "Negative" ? `${confidence}%` : "10%";
      }
      if (positivePercent) positivePercent.textContent = (sentiment === "Positive" ? confidence : 10) + "%";
      if (neutralPercent) neutralPercent.textContent = (sentiment === "Neutral" ? confidence : 10) + "%";
      if (negativePercent) negativePercent.textContent = (sentiment === "Negative" ? confidence : 10) + "%";

      if (sentimentStatus) {
        sentimentStatus.textContent = `Analyzed from: ${data.source} · ${data.char_count} characters`;
        sentimentStatus.dataset.state = "success";
      }
    } catch (error) {
      if (sentimentStatus) {
        sentimentStatus.textContent = `❌ Error: ${error.message}`;
        sentimentStatus.dataset.state = "error";
      }
    } finally {
      sentimentAnalyzeBtn.disabled = false;
      sentimentAnalyzeBtn.textContent = "Analyze";
    }
  });
}
