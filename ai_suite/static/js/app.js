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
