// ============================================================
// Plant Disease Detection System — Frontend JavaScript
// ============================================================
// This file handles all user interactions:
// 1. Drag & drop / file picker for image upload
// 2. Image preview before submission
// 3. Sending image to Flask backend via Fetch API
// 4. Displaying prediction results with animations
// ============================================================

// ---- Get references to DOM elements ----
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const previewArea   = document.getElementById('preview-area');
const previewImg    = document.getElementById('preview-img');
const cardActions   = document.getElementById('card-actions');
const btnAnalyze    = document.getElementById('btn-analyze');
const btnReset      = document.getElementById('btn-reset');
const btnChange     = document.getElementById('btn-change');
const btnTryAnother = document.getElementById('btn-try-another');
const loadingOverlay  = document.getElementById('loading-overlay');
const loadingStep     = document.getElementById('loading-step');
const resultSection   = document.getElementById('result-section');
const uploadSection   = document.getElementById('upload-section');

// Result display elements
const resultDiseaseName = document.getElementById('result-disease-name');
const resultIcon        = document.getElementById('result-icon');
const confidenceValue   = document.getElementById('confidence-value');
const confidenceBar     = document.getElementById('confidence-bar');
const resultDescription = document.getElementById('result-description');
const severityBadge     = document.getElementById('severity-badge');
const tipsList          = document.getElementById('tips-list');
const scoresList        = document.getElementById('scores-list');
const resultImg         = document.getElementById('result-img');

// ---- State ----
let selectedFile = null;  // Stores the currently selected image file

// ============================================================
// IMAGE SELECTION — File Picker
// ============================================================

// When user clicks the drop zone, trigger the hidden file input
dropZone.addEventListener('click', () => fileInput.click());

// When a file is chosen via file picker
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelected(file);
});

// "Change Image" button in the preview overlay
btnChange.addEventListener('click', () => {
    fileInput.click();
});

// ============================================================
// DRAG AND DROP SUPPORT
// ============================================================

// Visual feedback when dragging over the drop zone
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();              // Required to allow drop
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

// Handle file drop
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    const file = e.dataTransfer.files[0];
    if (file && isValidImageFile(file)) {
        handleFileSelected(file);
    } else {
        alert('Please drop a valid image file (JPG or PNG).');
    }
});

// ============================================================
// FILE HANDLING FUNCTIONS
// ============================================================

/**
 * Check if file is a valid image type.
 * @param {File} file
 * @returns {boolean}
 */
function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    return validTypes.includes(file.type);
}

/**
 * Handle a selected file:
 * - Store file reference
 * - Show image preview
 * - Show action buttons
 */
function handleFileSelected(file) {
    if (!isValidImageFile(file)) {
        alert('Invalid file type. Please use JPG or PNG.');
        return;
    }

    selectedFile = file;

    // Create a local URL for the image preview (doesn't upload yet)
    const objectURL = URL.createObjectURL(file);
    previewImg.src = objectURL;

    // Show preview, hide drop zone, show buttons
    dropZone.style.display = 'none';
    previewArea.style.display = 'block';
    cardActions.style.display = 'flex';
}

// ============================================================
// RESET FUNCTIONALITY
// ============================================================

/**
 * Reset the UI back to initial state.
 */
function resetUI() {
    selectedFile = null;
    fileInput.value = '';           // Clear file input
    previewImg.src = '';
    previewArea.style.display = 'none';
    cardActions.style.display = 'none';
    dropZone.style.display = 'block';
    resultSection.style.display = 'none';
    uploadSection.style.display = 'block';

    // Scroll back to upload section
    uploadSection.scrollIntoView({ behavior: 'smooth' });
}

// Wire up reset buttons
btnReset.addEventListener('click', resetUI);
btnTryAnother.addEventListener('click', resetUI);

// ============================================================
// LOADING ANIMATION
// ============================================================
// Show different messages to entertain the user during prediction

const loadingMessages = [
    'Preprocessing image…',
    'Extracting features…',
    'Running CNN model…',
    'Interpreting results…'
];

let loadingInterval = null;
let loadingMsgIndex = 0;

function startLoadingAnimation() {
    loadingMsgIndex = 0;
    loadingStep.textContent = loadingMessages[0];
    loadingOverlay.style.display = 'flex';

    // Cycle through messages every 800ms
    loadingInterval = setInterval(() => {
        loadingMsgIndex = (loadingMsgIndex + 1) % loadingMessages.length;
        loadingStep.textContent = loadingMessages[loadingMsgIndex];
    }, 800);
}

function stopLoadingAnimation() {
    clearInterval(loadingInterval);
    loadingOverlay.style.display = 'none';
}

// ============================================================
// PREDICTION — Send Image to Backend
// ============================================================

btnAnalyze.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select an image first.');
        return;
    }

    // Prepare form data (multipart/form-data for file upload)
    const formData = new FormData();
    formData.append('image', selectedFile);

    // Show loading animation
    startLoadingAnimation();

    try {
        // POST to Flask backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
            // NOTE: Do NOT set Content-Type header manually —
            //       the browser sets it with boundary automatically for FormData
        });

        // Check for HTTP errors
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || 'Server error occurred');
        }

        // Parse JSON response
        const result = await response.json();

        // Stop loading and show results
        stopLoadingAnimation();
        displayResults(result);

    } catch (error) {
        stopLoadingAnimation();
        console.error('Prediction error:', error);
        alert(`Error: ${error.message}\n\nMake sure the Flask server is running.`);
    }
});

// ============================================================
// DISPLAY RESULTS
// ============================================================

/**
 * Populate and show the result section with prediction data.
 * @param {Object} result - Prediction data from Flask backend
 */
function displayResults(result) {
    // 1. Set disease name and icon
    resultDiseaseName.textContent = result.disease;
    resultIcon.textContent = getDiseaseIcon(result.disease);

    // 2. Set confidence percentage
    confidenceValue.textContent = `${result.confidence}%`;

    // 3. Animate confidence bar (delay for CSS transition)
    setTimeout(() => {
        confidenceBar.style.width = `${result.confidence}%`;

        // Change bar color based on confidence level
        if (result.confidence >= 85) {
            confidenceBar.style.background = 'linear-gradient(90deg, #15803d, #22c55e)';
        } else if (result.confidence >= 65) {
            confidenceBar.style.background = 'linear-gradient(90deg, #b45309, #f59e0b)';
        } else {
            confidenceBar.style.background = 'linear-gradient(90deg, #991b1b, #ef4444)';
        }
    }, 100);

    // 4. Disease description
    resultDescription.textContent = result.description;

    // 5. Severity badge
    severityBadge.textContent = result.severity === 'none' ? 'No disease' : result.severity;
    severityBadge.className = `severity-badge ${result.severity}`;

    // 6. Treatment tips — build list items
    tipsList.innerHTML = '';
    (result.tips || []).forEach((tip) => {
        const li = document.createElement('li');
        li.textContent = tip;
        tipsList.appendChild(li);
    });

    // 7. All class probability scores
    scoresList.innerHTML = '';
    const scores = result.all_scores || {};
    const topClass = result.disease;

    Object.entries(scores)
        .sort((a, b) => b[1] - a[1])  // Sort by score descending
        .forEach(([className, score]) => {
            const isTop = className === topClass;
            const row = document.createElement('div');
            row.className = 'score-row';
            row.innerHTML = `
                <div class="score-header">
                    <span>${className}</span>
                    <span>${score.toFixed(1)}%</span>
                </div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill ${isTop ? 'is-top' : ''}"
                         style="width: 0%"
                         data-width="${score}">
                    </div>
                </div>
            `;
            scoresList.appendChild(row);
        });

    // Animate score bars after DOM is painted
    requestAnimationFrame(() => {
        setTimeout(() => {
            document.querySelectorAll('.score-bar-fill').forEach((bar) => {
                bar.style.width = `${bar.dataset.width}%`;
            });
        }, 200);
    });

    // 8. Show result image
    resultImg.src = result.image_url + '?t=' + Date.now(); // Cache-bust

    // 9. Show demo mode warning if model not loaded
    if (result.demo_mode) {
        const demoBanner = document.createElement('div');
        demoBanner.className = 'demo-banner';
        demoBanner.textContent = '⚠️ Demo mode — random prediction. Run train_model.py to train the real model.';
        resultSection.querySelector('.result-footer').prepend(demoBanner);
    }

    // 10. Hide upload section, show results
    uploadSection.style.display = 'none';
    resultSection.style.display = 'block';

    // Scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// ============================================================
// HELPER — Get emoji icon for each disease class
// ============================================================

/**
 * Returns an emoji representing the disease.
 * @param {string} disease
 * @returns {string}
 */
function getDiseaseIcon(disease) {
    const icons = {
        'Healthy':      '🌿',
        'Early Blight': '🍂',
        'Late Blight':  '🍁',
        'Leaf Mold':    '🍃'
    };
    return icons[disease] || '🌱';
}

// ============================================================
// PAGE LOAD — Initialization
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('[PlantScan] Disease Detection System loaded.');
    console.log('[PlantScan] Upload a leaf image to get started.');
});
