const KEYS = { chats: "study-companion-chats-v1", preferences: "study-companion-preferences-v1" };
const $ = (selector) => document.querySelector(selector);
const ui = {
  form: $("#chat-form"), input: $("#question-input"), send: $("#send-button"), messages: $("#message-list"),
  welcome: $("#welcome"), scroll: $("#chat-scroll"), chats: $("#conversation-list"), count: $("#conversation-count"),
  title: $("#active-title"), file: $("#document-input"), docs: $("#document-list"), uploadStatus: $("#upload-status"),
  selected: $("#selected-documents"), connection: $("#connection-status"), dialog: $("#settings-dialog"),
  settings: $("#settings-form"), sidebar: $("#sidebar"), scrim: $("#sidebar-scrim"),
};
function load(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } }
let chats = load(KEYS.chats, []);
let prefs = load(KEYS.preferences, { name: "", style: "", goals: "" });
let activeId = chats[0]?.id ?? null;
let docs = [];
let selectedIds = new Set();
let sending = false;
function save() {
  localStorage.setItem(KEYS.chats, JSON.stringify(chats));
  localStorage.setItem(KEYS.preferences, JSON.stringify(prefs));
}
function currentChat() { return chats.find((chat) => chat.id === activeId) ?? null; }
function newChat() {
  const chat = { id: crypto.randomUUID(), title: "New conversation", updatedAt: Date.now(), messages: [] };
  chats.unshift(chat); activeId = chat.id; save(); render(); ui.input.focus(); closeSidebar();
}
function cleanTitle(value) { return value.trim().replace(/\s+/g, " ").slice(0, 38) || "New conversation"; }
function renderChats() {
  ui.chats.replaceChildren(); ui.count.textContent = String(chats.length);
  for (const chat of [...chats].sort((a, b) => b.updatedAt - a.updatedAt)) {
    const row = document.createElement("div"); row.className = `conversation-item${chat.id === activeId ? " active" : ""}`;
    const open = document.createElement("button"); open.className = "conversation-open"; open.type = "button";
    open.setAttribute("aria-label", `Open ${chat.title}`);
    open.innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 5.5h12M4 10h12M4 14.5h8"/></svg>';
    const title = document.createElement("span"); title.className = "conversation-name"; title.textContent = chat.title; open.append(title);
    open.addEventListener("click", () => { activeId = chat.id; render(); closeSidebar(); });
    const actions = document.createElement("span"); actions.className = "conversation-actions";
    const rename = document.createElement("button"); rename.className = "tiny-action"; rename.type = "button";
    rename.textContent = "✎"; rename.title = "Rename conversation"; rename.setAttribute("aria-label", `Rename ${chat.title}`);
    rename.addEventListener("click", () => {
      const value = window.prompt("Conversation name", chat.title);
      if (value !== null) { chat.title = cleanTitle(value); save(); render(); }
    });
    const remove = document.createElement("button"); remove.className = "tiny-action"; remove.type = "button";
    remove.textContent = "×"; remove.title = "Delete conversation"; remove.setAttribute("aria-label", `Delete ${chat.title}`);
    remove.addEventListener("click", () => {
      if (!window.confirm(`Delete “${chat.title}”?`)) return;
      chats = chats.filter((item) => item.id !== chat.id);
      if (activeId === chat.id) activeId = chats[0]?.id ?? null;
      save(); render();
    });
    actions.append(rename, remove); row.append(open, actions); ui.chats.append(row);
  }
}
function makeMessage(message) {
  const row = document.createElement("article"); row.className = `message ${message.role}${message.error ? " error" : ""}`;
  if (message.role === "assistant") {
    const avatar = document.createElement("span"); avatar.className = "message-avatar"; avatar.textContent = "✳";
    avatar.setAttribute("aria-hidden", "true"); row.append(avatar);
  }
  const wrap = document.createElement("div"); wrap.className = "message-content-wrap";
  const body = document.createElement("div"); body.className = "message-content"; body.textContent = message.content; wrap.append(body);
  if (message.role === "assistant" && !message.error) {
    const tools = document.createElement("div"); tools.className = "message-tools";
    if (message.sources?.length) {
      const source = document.createElement("span"); source.className = "source-note";
      source.textContent = `Based on uploaded documents: ${message.sources.join(", ")}`; tools.append(source);
    }
    const copy = document.createElement("button"); copy.className = "copy-button"; copy.type = "button";
    copy.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="5" y="5" width="8" height="9" rx="1"/><path d="M10.5 3H4a1 1 0 0 0-1 1v7"/></svg><span>Copy</span>';
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(message.content); copy.querySelector("span").textContent = "Copied"; }
      catch { copy.querySelector("span").textContent = "Copy unavailable"; }
      window.setTimeout(() => { copy.querySelector("span").textContent = "Copy"; }, 1400);
    });
    tools.append(copy); wrap.append(tools);
  }
  row.append(wrap); return row;
}
function renderMessages() {
  const chat = currentChat(); const messages = chat?.messages ?? [];
  ui.welcome.hidden = messages.length > 0; ui.messages.replaceChildren(...messages.map(makeMessage));
  ui.title.textContent = chat?.title ?? "New conversation";
  ui.scroll.scrollTop = messages.length ? ui.scroll.scrollHeight : 0;
}
function renderDocs() {
  ui.docs.replaceChildren(); ui.selected.replaceChildren();
  if (!docs.length) { const empty = document.createElement("p"); empty.className = "muted-empty"; empty.textContent = "No documents uploaded"; ui.docs.append(empty); }
  for (const file of docs) {
    const row = document.createElement("div"); row.className = "document-row";
    const checkbox = document.createElement("input"); checkbox.className = "document-checkbox"; checkbox.type = "checkbox";
    checkbox.checked = selectedIds.has(file.id); checkbox.setAttribute("aria-label", `Use ${file.name} in chat`);
    checkbox.addEventListener("change", () => { checkbox.checked ? selectedIds.add(file.id) : selectedIds.delete(file.id); renderDocs(); });
    const type = document.createElement("span"); type.className = "document-type"; type.textContent = file.name.split(".").pop().slice(0, 4).toUpperCase();
    const name = document.createElement("span"); name.className = "document-name"; name.textContent = file.name; name.title = file.name;
    const remove = document.createElement("button"); remove.className = "tiny-action remove-document"; remove.type = "button";
    remove.textContent = "×"; remove.title = "Remove document"; remove.setAttribute("aria-label", `Remove ${file.name}`);
    remove.addEventListener("click", () => removeDoc(file)); row.append(checkbox, type, name, remove); ui.docs.append(row);
    if (selectedIds.has(file.id)) { const chip = document.createElement("span"); chip.className = "selected-chip"; chip.textContent = `Using: ${file.name}`; ui.selected.append(chip); }
  }
}
function render() {
  renderChats(); renderMessages(); renderDocs();
  const label = prefs.name.trim() || "Study preferences";
  $("#profile-name").textContent = label; $("#profile-avatar").textContent = label === "Study preferences" ? "S" : label[0].toUpperCase();
}
function uploadStatus(message, error = false) { ui.uploadStatus.textContent = message; ui.uploadStatus.classList.toggle("error", error); }
async function refreshDocs() {
  try {
    const response = await fetch("/api/documents"); const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load documents.");
    docs = data.documents; const existing = new Set(docs.map((item) => item.id));
    selectedIds = new Set([...selectedIds].filter((id) => existing.has(id))); renderDocs();
  } catch (error) { uploadStatus(error.message, true); }
}
async function uploadFiles(files) {
  for (const file of files) {
    const extension = file.name.split(".").pop().toLowerCase();
    if (!["pdf", "docx", "txt"].includes(extension)) { uploadStatus(`${file.name}: choose a PDF, DOCX, or TXT file.`, true); continue; }
    if (file.size > 10 * 1024 * 1024) { uploadStatus(`${file.name}: files must be 10 MB or smaller.`, true); continue; }
    uploadStatus(`Processing ${file.name}…`); const form = new FormData(); form.append("file", file);
    try {
      const response = await fetch("/api/documents", { method: "POST", body: form }); const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Document upload failed.");
      docs.push(data); selectedIds.add(data.id); uploadStatus(`${data.name} is ready for document-grounded questions.`); renderDocs();
    } catch (error) { uploadStatus(error.message, true); }
  }
  ui.file.value = "";
}
async function removeDoc(file) {
  try {
    const response = await fetch(`/api/documents/${encodeURIComponent(file.id)}`, { method: "DELETE" }); const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not remove document.");
    docs = docs.filter((item) => item.id !== file.id); selectedIds.delete(file.id); uploadStatus(`${file.name} removed.`); renderDocs();
  } catch (error) { uploadStatus(error.message, true); }
}
function typing() {
  const row = document.createElement("div"); row.className = "message assistant"; row.id = "typing-indicator";
  const avatar = document.createElement("span"); avatar.className = "message-avatar"; avatar.textContent = "✳";
  avatar.setAttribute("aria-hidden", "true"); const dots = document.createElement("span"); dots.className = "typing-dots";
  dots.setAttribute("aria-label", "Assistant is generating a response"); dots.innerHTML = "<i></i><i></i><i></i>";
  row.append(avatar, dots); ui.messages.append(row); ui.scroll.scrollTop = ui.scroll.scrollHeight;
}
function resizeInput() { ui.input.style.height = "auto"; ui.input.style.height = `${Math.min(ui.input.scrollHeight, 150)}px`; }
async function sendQuestion(value = ui.input.value) {
  const question = value.trim(); if (!question || sending) return;
  if (!currentChat()) newChat(); const chat = currentChat();
  chat.messages.push({ role: "user", content: question });
  if (chat.messages.filter((item) => item.role === "user").length === 1) chat.title = cleanTitle(question);
  chat.updatedAt = Date.now(); save(); ui.input.value = ""; resizeInput(); render(); sending = true; ui.send.disabled = true; typing();
  try {
    const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: chat.messages.slice(0, -1).slice(-12), preferences: prefs,
        document_ids: [...selectedIds], conversation_id: chat.id }) });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || "The assistant could not respond.");
    chat.messages.push({ role: "assistant", content: result.answer, sources: result.sources ?? [] });
  } catch (error) {
    chat.messages.push({ role: "assistant", content: error.message || "Could not connect to the assistant. Check your connection and try again.", error: true });
  } finally { sending = false; ui.send.disabled = false; save(); render(); ui.input.focus(); }
}
async function checkHealth() {
  try {
    const response = await fetch("/api/health"); const result = await response.json(); if (!response.ok) throw new Error(result.message);
    ui.connection.querySelector(".status-dot").className = "status-dot"; ui.connection.lastElementChild.textContent = "Model connected";
  } catch {
    ui.connection.querySelector(".status-dot").className = "status-dot offline"; ui.connection.lastElementChild.textContent = "Model unavailable";
  }
}
function closeSidebar() { ui.sidebar.classList.remove("sidebar-open"); ui.scrim.classList.remove("scrim-open"); }
$("#new-chat-button").addEventListener("click", newChat);
ui.form.addEventListener("submit", (event) => { event.preventDefault(); sendQuestion(); });
ui.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); sendQuestion(); }
});
ui.input.addEventListener("input", resizeInput);
$("#upload-trigger").addEventListener("click", () => ui.file.click());
ui.file.addEventListener("change", () => uploadFiles([...ui.file.files]));
document.querySelectorAll(".suggestion").forEach((button) => button.addEventListener("click", () => sendQuestion(button.dataset.prompt)));
$("#settings-button").addEventListener("click", () => {
  ui.settings.elements.name.value = prefs.name; ui.settings.elements.style.value = prefs.style;
  ui.settings.elements.goals.value = prefs.goals; ui.dialog.showModal();
});
ui.settings.addEventListener("submit", () => {
  prefs = { name: ui.settings.elements.name.value.trim(), style: ui.settings.elements.style.value,
    goals: ui.settings.elements.goals.value.trim() }; save(); render();
});
$("#menu-button").addEventListener("click", () => { ui.sidebar.classList.add("sidebar-open"); ui.scrim.classList.add("scrim-open"); });
ui.scrim.addEventListener("click", closeSidebar);
window.addEventListener("storage", (event) => {
  if (event.key === KEYS.chats) chats = load(KEYS.chats, []);
  if (event.key === KEYS.preferences) prefs = load(KEYS.preferences, prefs);
  render();
});
render(); refreshDocs(); checkHealth();
