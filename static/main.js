document.getElementById("predictBtn").addEventListener("click", async () => {
  const lyrics = document.getElementById("lyrics").value;
  if (!lyrics.trim()) {
    alert("Ingresa la letra de la canción.");
    return;
  }

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lyrics }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert("Error: " + (err.error || res.statusText));
    return;
  }

  const data = await res.json();
  document.getElementById("output").style.display = "block";
  document.getElementById("predLabel").innerText = data.prediction;
  document.getElementById("predIndex").innerText = `Index: ${data.pred_index}`;

  const probsDiv = document.getElementById("probs");
  probsDiv.innerHTML = "";
  const probs = data.probs || {};
  // ordenar por prob desc
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  for (const [label, prob] of entries) {
    const div = document.createElement("div");
    div.className = "prob-item";
    div.innerHTML = `<span>${label}</span><span>${(prob * 100).toFixed(
      2
    )}%</span>`;
    probsDiv.appendChild(div);
  }
});

document.getElementById("clearBtn").addEventListener("click", () => {
  document.getElementById("lyrics").value = "";
  document.getElementById("output").style.display = "none";
});
