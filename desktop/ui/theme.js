try {
  if (localStorage.getItem("redpanda-theme") === "dark") {
    document.documentElement.dataset.theme = "dark";
  }
} catch {}
