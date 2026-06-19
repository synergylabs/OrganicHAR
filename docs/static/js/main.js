/* VAX project page — minimal vanilla JS: carousel + copy buttons. No dependencies. */
(function () {
  "use strict";

  /* ---------- Copy-to-clipboard for code blocks ---------- */
  document.querySelectorAll(".copy-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var sel = btn.getAttribute("data-target");
      var node = document.getElementById(sel);
      if (!node) return;
      var text = node.innerText.replace(/\n+$/, "");
      navigator.clipboard.writeText(text).then(function () {
        var old = btn.textContent;
        btn.textContent = "Copied ✓";
        btn.classList.add("ok");
        setTimeout(function () { btn.textContent = old; btn.classList.remove("ok"); }, 1600);
      });
    });
  });

  /* ---------- Carousel ---------- */
  document.querySelectorAll("[data-carousel]").forEach(function (car) {
    var slides = Array.prototype.slice.call(car.querySelectorAll(".slide"));
    var dots = Array.prototype.slice.call(car.querySelectorAll(".dot"));
    var counter = car.querySelector(".counter");
    var i = 0;

    function show(n) {
      i = (n + slides.length) % slides.length;
      slides.forEach(function (s, k) { s.classList.toggle("active", k === i); });
      dots.forEach(function (d, k) { d.classList.toggle("active", k === i); });
      if (counter) counter.textContent = (i + 1) + " / " + slides.length;
    }
    var prev = car.querySelector(".prev");
    var next = car.querySelector(".next");
    if (prev) prev.addEventListener("click", function () { show(i - 1); });
    if (next) next.addEventListener("click", function () { show(i + 1); });
    dots.forEach(function (d, k) { d.addEventListener("click", function () { show(k); }); });

    car.setAttribute("tabindex", "0");
    car.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") show(i - 1);
      if (e.key === "ArrowRight") show(i + 1);
    });
    show(0);
  });

  /* ---------- Active nav highlight ---------- */
  var navLinks = Array.prototype.slice.call(document.querySelectorAll(".nav .links a"));
  var sections = navLinks
    .map(function (a) { return document.querySelector(a.getAttribute("href")); })
    .filter(Boolean);
  if ("IntersectionObserver" in window && sections.length) {
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          var id = "#" + en.target.id;
          navLinks.forEach(function (a) {
            a.style.color = a.getAttribute("href") === id ? "var(--accent)" : "";
          });
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    sections.forEach(function (s) { obs.observe(s); });
  }
})();
