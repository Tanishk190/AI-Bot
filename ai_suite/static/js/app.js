const panels = document.querySelectorAll(".panel");
const navItems = document.querySelectorAll(".nav-item");
const pageTitle = document.querySelector("#page-title");
const pageBadge = document.querySelector("#page-badge");

function showPanel(item) {
  const id = item.dataset.panel;

  panels.forEach((panel) => panel.classList.remove("active"));
  navItems.forEach((navItem) => navItem.classList.remove("active"));

  document.querySelector(`#panel-${id}`).classList.add("active");
  item.classList.add("active");
  pageTitle.textContent = item.dataset.title;
  pageBadge.textContent = item.dataset.badge;
}

navItems.forEach((item) => {
  item.addEventListener("click", () => showPanel(item));
});

const chatDocuments = document.querySelector("#chat-documents");
const chatIndexButton = document.querySelector("#chat-index-btn");
const chatIndexStatus = document.querySelector("#chat-index-status");
const chatHistory = document.querySelector("#chat-history");
const chatForm = document.querySelector("#chat-form");
const chatInput = document.querySelector("#chat-input");

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
    Array.from(chatDocuments.files).forEach((file) => {
      formData.append("documents", file);
    });

    chatIndexButton.disabled = true;
    setChatStatus("Indexing documents...", "ready");

    try {
      const response = await fetch("/api/chat/index", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Document indexing failed.");
      }

      const filenames = data.files.map((file) => file.name).join(", ");
      setChatStatus(`Indexed ${data.total_chunks} chunks from ${filenames}.`, "success");
      appendMessage("ai", "Documents are indexed. Ask a question and I will answer from the retrieved context.");
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
    const pendingBubble = appendMessage("ai", "Searching Documents...");

    try {
      const response = await fetch("/api/chat/message", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Chat request failed.");
      }

      pendingBubble.textContent = data.answer || "No answer returned.";
    } catch (error) {
      pendingBubble.textContent = error.message;
    } finally {
      chatInput.disabled = false;
      chatInput.focus();
    }
  });
}

// PII Extractor
const piiSystemPrompt = document.querySelector("#pii-system-prompt");
const piiInputText = document.querySelector("#pii-input-text");
const piiExtractBtn = document.querySelector("#pii-extract-btn");
const piiOutput = document.querySelector("#pii-output");
const piiDownloadBtn = document.querySelector("#pii-download-btn");
const piiCopyBtn = document.querySelector("#pii-copy-btn");

let lastPiiData = null;

function normalizePiiEntities(data) {
  if (!data) return [];
  if (Array.isArray(data)) return data;
  if (typeof data === "object") {
    if (Array.isArray(data.entities)) return data.entities;
    const entries = Object.entries(data);
    const objectValues = entries.filter(([, value]) => value && typeof value === "object" && !Array.isArray(value));
    const entityLikeKeys = entries.filter(([key]) => /\d/.test(key) || /person|entity/i.test(key));
    if (objectValues.length === entries.length && entityLikeKeys.length >= Math.max(1, Math.ceil(entries.length / 2))) {
      return entries.map(([key, value]) => ({ entity: key, ...value }));
    }
    const arrayFields = entries.filter(([, value]) => Array.isArray(value));
    if (arrayFields.length && arrayFields.length === entries.length) {
      const lengths = arrayFields.map(([, value]) => value.length);
      const rowCount = Math.max(...lengths);
      if (lengths.every((length) => length === rowCount)) {
        return Array.from({ length: rowCount }, (_, index) => {
          const row = {};
          arrayFields.forEach(([key, values]) => {
            row[key] = values[index];
          });
          return row;
        });
      }
    }
    const arrayValues = arrayFields.map(([, value]) => value);
    if (arrayValues.length === 1) return arrayValues[0];
    return [data];
  }
  return [data];
}

function formatPiiCell(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "object") {
    const jsonValue = JSON.stringify(value);
    return jsonValue === undefined ? String(value) : jsonValue;
  }
  return String(value);
}

function renderPiiTable(data) {
  if (!piiOutput) return;
  const entities = normalizePiiEntities(data);
  if (!entities.length) {
    piiOutput.innerHTML = '<div class="table-empty">No PII entities found.</div>';
    return;
  }

  const normalized = entities.map((entry) => {
    if (entry && typeof entry === "object" && !Array.isArray(entry)) {
      return entry;
    }
    return { value: entry };
  });

  const columns = [];
  normalized.forEach((entry) => {
    Object.keys(entry).forEach((key) => {
      if (!columns.includes(key)) {
        columns.push(key);
      }
    });
  });

  if (!columns.length) {
    columns.push("value");
  }

  const table = document.createElement("table");
  table.className = "pii-table";
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column;
    headerRow.append(th);
  });
  thead.append(headerRow);

  const tbody = document.createElement("tbody");
  normalized.forEach((entry) => {
    const row = document.createElement("tr");
    columns.forEach((column) => {
      const cell = document.createElement("td");
      cell.textContent = formatPiiCell(entry[column]);
      row.append(cell);
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

    if (!text) {
      piiOutput.textContent = "❌ Error: Input text is required.";
      return;
    }

    if (!systemPrompt) {
      piiOutput.textContent = "❌ Error: System prompt is required.";
      return;
    }

    piiExtractBtn.disabled = true;
    piiOutput.textContent = "⏳ Extracting PII...";

    try {
      const response = await fetch("/api/pii/extract", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: text,
          system_prompt: systemPrompt,
        }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "PII extraction failed.");
      }

      let piiData = data.data;
      if (typeof piiData === "string") {
        piiData = JSON.parse(piiData);
      }
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
    if (!lastPiiData) {
      alert("Extract PII first");
      return;
    }
    const text = JSON.stringify(lastPiiData, null, 2);
    navigator.clipboard.writeText(text).then(() => {
      piiCopyBtn.textContent = "✓ Copied!";
      setTimeout(() => {
        piiCopyBtn.textContent = "Copy JSON";
      }, 2000);
    });
  });
}

if (piiDownloadBtn) {
  piiDownloadBtn.addEventListener("click", () => {
    if (!lastPiiData) {
      alert("Extract PII first");
      return;
    }
    const json = JSON.stringify(lastPiiData, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "pii_extraction.json";
    a.click();
    URL.revokeObjectURL(url);
  });
}
