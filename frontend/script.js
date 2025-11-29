
// frontend/script.js

const API_BASE = "http://127.0.0.1:8000";

let imageId = null;
let catalog = [];

const statusEl = document.getElementById("status");
const regionSelect = document.getElementById("regionSelect");
const tileButtonsDiv = document.getElementById("tileButtons");
const controlsDiv = document.getElementById("controls");
const previewImg = document.getElementById("preview");
const uploadBtn = document.getElementById("uploadBtn");
const roomInput = document.getElementById("roomInput");

uploadBtn.addEventListener("click", uploadRoom);

async function uploadRoom() {
  if (!roomInput.files || roomInput.files.length === 0) {
    alert("Please select a room image first.");
    return;
  }

  statusEl.textContent = "Uploading and segmenting...";
  imageId = null;
  previewImg.src = "";

  const formData = new FormData();
  formData.append("image", roomInput.files[0]);

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }

    const data = await res.json();
    imageId = data.image_id;
    statusEl.textContent = `Image uploaded. image_id = ${imageId}`;

    // Show original image from local file as initial preview
    const fileURL = URL.createObjectURL(roomInput.files[0]);
    previewImg.src = fileURL;

    // Load catalogue & show controls
    await loadCatalog();
    controlsDiv.style.display = "block";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error: " + err.message;
  }
}

async function loadCatalog() {
  try {
    const res = await fetch(`${API_BASE}/catalog`);
    if (!res.ok) {
      throw new Error("Failed to load catalog");
    }
    catalog = await res.json();
    renderTileButtons();
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error loading catalog: " + err.message;
  }
}

function renderTileButtons() {
  tileButtonsDiv.innerHTML = "";
  catalog.forEach((tile) => {
    const btn = document.createElement("button");
    btn.textContent = tile.name;
    btn.addEventListener("click", () =>
      applyTexture(regionSelect.value, tile.id)
    );
    tileButtonsDiv.appendChild(btn);
  });
}

async function applyTexture(region, textureId) {
  if (!imageId) {
    alert("No image uploaded yet.");
    return;
  }

  statusEl.textContent = `Applying ${textureId} on ${region}...`;

  try {
    const url = `${API_BASE}/apply_texture?image_id=${encodeURIComponent(
      imageId
    )}&region=${encodeURIComponent(region)}&texture_id=${encodeURIComponent(
      textureId
    )}&t=${Date.now()}`;

    // Just set the img src to the API URL
    previewImg.src = url;
    statusEl.textContent = "Done.";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error: " + err.message;
  }
}
