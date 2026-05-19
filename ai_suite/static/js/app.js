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
