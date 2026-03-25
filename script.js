
document.addEventListener('DOMContentLoaded', () => {
  const stage = document.querySelector('[data-stack]');
  if (!stage) return;
  const cards = Array.from(stage.querySelectorAll('.stack-card'));
  const counter = document.querySelector('[data-stack-counter]');
  let start = 0;

  function render() {
    cards.forEach((card, i) => {
      const rel = (i - start + cards.length) % cards.length;
      card.dataset.pos = rel <= 4 ? String(rel) : '5';
      card.style.display = rel <= 4 ? 'block' : 'none';
    });
    if (counter) counter.textContent = `${String(start + 1).padStart(2,'0')} / ${String(cards.length).padStart(2,'0')}`;
  }

  function next() { start = (start + 1) % cards.length; render(); }
  function prev() { start = (start - 1 + cards.length) % cards.length; render(); }

  document.querySelector('[data-stack-next]')?.addEventListener('click', next);
  document.querySelector('[data-stack-prev]')?.addEventListener('click', prev);
  render();
  setInterval(next, 3200);
});
