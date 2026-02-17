// Application State
const state = {
    embeddingsFolder: '',
    imagesFolder: '',
    subfolders: [],
    weights: Array(12).fill(1 / 12),
    weightBounds: Array(12).fill([0.0, 1.0]),
    similarityMatrix: null,
    groundTruthMatrix: null,
    comparisonMatrix: null,
    labels: [],
    embeddingLabels: [],  // Labels from filenames
    tsneData: null,
    selectedPoint: null,
    tsneImages: [], // For image iteration
    tsneImageIndex: 0,
    lastImageIndex: null, // Memory for image selection
    slots: [], // For comparison slots (max 24)
    isDarkMode: window.DARK_MODE || false,
    preventGTReorder: false,
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeWeightSliders();
    setupDarkMode();
    setupTSNEVisibility();
    setupEventListeners();
});

function setupTSNEVisibility() {
    if (window.HIDE_TSNE) {
        const tsneSection = document.getElementById('tsne-section');
        if (tsneSection) {
            tsneSection.classList.add('hidden');
        }
    }
}

function setupEventListeners() {
    const gtReorderCheckbox = document.getElementById('prevent-gt-reorder');
    if (gtReorderCheckbox) {
        gtReorderCheckbox.addEventListener('change', (e) => {
            state.preventGTReorder = e.target.checked;
        });
    }
}

function setupDarkMode() {
    if (state.isDarkMode) {
        document.body.classList.add('dark');
    }
}

// Folder Operations
async function loadSubfolders() {
    const folder = document.getElementById('embeddings-folder').value.trim();
    if (!folder) return alert('Please enter embeddings folder path');

    state.embeddingsFolder = folder;

    try {
        const response = await fetch(`/api/subfolders?folder=${encodeURIComponent(folder)}`);
        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        state.subfolders = data.subfolders;
        state.labels = data.subfolders;

        // Store embedding labels if available
        if (data.embedding_labels && data.embedding_labels.length > 0) {
            state.embeddingLabels = data.embedding_labels;
            initializeWeightSliders();  // Re-init with new labels
            initializeTsneCheckboxes();
        }

        // Update UI with info
        document.getElementById('folder-info').classList.remove('hidden');
        document.getElementById('subfolders-count').textContent = state.subfolders.length;

        if (data.info) {
            document.getElementById('images-per-subfolder').textContent = data.info.n_images || 12;
            document.getElementById('embedding-dim').textContent = data.info.embedding_dim || 512;
        }

        // Prepopulate custom axis order field
        const axisOrderInput = document.getElementById('custom-axis-order');
        if (axisOrderInput) {
            axisOrderInput.value = state.labels.join(', ');
        }

        // Populate dropdowns
        populateSubfolderDropdowns();

        // Initialize slider labels
        if (!state.embeddingLabels.length) {
            initializeWeightSliders();
            initializeTsneCheckboxes();
        }
    } catch (error) {
        alert(`Error loading components: ${error.message}`);
    }
}

async function setImagesFolder() {
    const folder = document.getElementById('images-folder').value.trim();
    if (!folder) return;

    state.imagesFolder = folder;

    try {
        const response = await fetch(`/api/validate-images-folder?folder=${encodeURIComponent(folder)}`);
        const data = await response.json();

        if (data.error) {
            document.getElementById('images-info').classList.add('hidden');
            alert(data.error);
            return;
        }

        // Update state with labels from images
        if (data.embedding_labels && data.embedding_labels.length > 0) {
            state.embeddingLabels = data.embedding_labels;
            initializeWeightSliders();
            initializeTsneCheckboxes();
        }

        // Show feedback
        document.getElementById('images-info').classList.remove('hidden');
        document.getElementById('images-components').textContent = data.n_components;
        document.getElementById('images-count').textContent = data.n_images_per_component;

        // Update custom axis order if we have new labels
        if (data.embedding_labels && data.embedding_labels.length > 0) {
            const axisOrderInput = document.getElementById('custom-axis-order');
            if (axisOrderInput) {
                axisOrderInput.value = state.labels.join(', ');
            }
        }
    } catch (error) {
        alert(`Error validating images folder: ${error.message}`);
    }
}

async function loadGroundTruth() {
    const path = document.getElementById('ground-truth-path').value.trim();
    if (!path) return alert('Please enter ground truth CSV path');

    try {
        const response = await fetch(`/api/load-csv?path=${encodeURIComponent(path)}`);
        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        state.groundTruthMatrix = data.matrix;
        document.getElementById('gt-info').classList.remove('hidden');
        document.getElementById('gt-size').textContent = data.shape[0];
        document.getElementById('gt-size2').textContent = data.shape[1] || data.shape[0];

        // Re-render matrices if visible
        renderSideBySideMatrices();
    } catch (error) {
        alert(`Error loading ground truth: ${error.message}`);
    }
}

async function loadWeights() {
    const path = document.getElementById('weights-path').value.trim();
    if (!path) return alert('Please enter weights CSV path');

    try {
        const response = await fetch(`/api/load-csv?path=${encodeURIComponent(path)}`);
        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        const weights = Array.isArray(data.matrix[0]) ? data.matrix[0] : data.matrix;

        if (weights.length !== 12) {
            alert(`Expected 12 weights, got ${weights.length}`);
            return;
        }

        state.weights = weights;
        updateWeightSliders(weights);

        document.getElementById('weights-info').classList.remove('hidden');
        document.getElementById('weights-count').textContent = weights.length;
    } catch (error) {
        alert(`Error loading weights: ${error.message}`);
    }
}

function populateSubfolderDropdowns() {
    const leftSelect = document.getElementById('left-subfolder');
    const rightSelect = document.getElementById('right-subfolder');
    const searchSelect = document.getElementById('search-subfolder');

    [leftSelect, rightSelect, searchSelect].forEach(select => {
        if (!select) return;
        select.innerHTML = '<option value="">-- Select Component --</option>';
        state.subfolders.forEach(subfolder => {
            const option = document.createElement('option');
            option.value = subfolder;
            option.textContent = subfolder;
            select.appendChild(option);
        });
    });
}

// Weight Sliders
function initializeWeightSliders() {
    const container = document.getElementById('weight-sliders');
    container.innerHTML = '';

    for (let i = 0; i < 12; i++) {
        // Use embedding labels if available, otherwise use Weight X
        const label = state.embeddingLabels[i] || `Weight ${i}`;

        const div = document.createElement('div');
        div.className = 'weight-slider-container';
        div.innerHTML = `
            <div class="weight-slider-label" title="${label}">${label.slice(0, 12)}</div>
            <div class="weight-slider-row">
                <input type="range" id="weight-${i}" class="weight-slider" 
                    min="0" max="1" step="0.01" value="${state.weights[i]}"
                    oninput="onWeightChange(${i}, this.value)">
                <input type="text" id="weight-input-${i}" class="weight-input" 
                    value="${state.weights[i].toFixed(3)}"
                    onchange="onWeightInputChange(${i}, this.value)">
            </div>
        `;
        container.appendChild(div);
    }
    updateWeightSum();
}

function initializeTsneCheckboxes() {
    const container = document.getElementById('tsne-embedding-checkboxes');
    container.innerHTML = '';

    for (let i = 0; i < 12; i++) {
        const label = state.embeddingLabels[i] || `Embedding ${i}`;

        const div = document.createElement('label');
        div.className = 'flex items-center gap-1';
        div.innerHTML = `
            <input type="checkbox" id="tsne-embed-${i}" class="form-checkbox" checked>
            <span class="text-xs text-gray-600" title="${label}">${label.slice(0, 8)}</span>
        `;
        container.appendChild(div);
    }
}

function onWeightChange(index, value) {
    const numValue = parseFloat(value);
    state.weights[index] = numValue;
    document.getElementById(`weight-input-${index}`).value = numValue.toFixed(3);
    updateWeightSum();
}

function onWeightInputChange(index, value) {
    const numValue = parseFloat(value);
    if (!isNaN(numValue) && numValue >= 0 && numValue <= 1) {
        state.weights[index] = numValue;
        document.getElementById(`weight-${index}`).value = numValue;
        updateWeightSum();
    }
}

function onBoundChange(index, type, value) {
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
        if (type === 'min') {
            state.weightBounds[index][0] = numValue;
            document.getElementById(`weight-${index}`).min = numValue;
        } else {
            state.weightBounds[index][1] = numValue;
            document.getElementById(`weight-${index}`).max = numValue;
        }
    }
}

function updateWeightSliders(weights) {
    for (let i = 0; i < weights.length; i++) {
        document.getElementById(`weight-${i}`).value = weights[i];
        document.getElementById(`weight-input-${i}`).value = weights[i].toFixed(3);
    }
    updateWeightSum();
}

function updateWeightSum() {
    const sum = state.weights.reduce((a, b) => a + b, 0);
    const sumElement = document.getElementById('weight-sum');
    sumElement.textContent = sum.toFixed(3);

    // Color feedback
    if (Math.abs(sum - 1.0) < 0.01) {
        sumElement.className = 'text-green-600 font-medium';
    } else {
        sumElement.className = 'text-red-500 font-medium';
    }
}

function resetWeights() {
    state.weights = Array(12).fill(1 / 12);
    updateWeightSliders(state.weights);
}

function onMethodChange() {
    const method = document.getElementById('similarity-method').value;
    const weightsSection = document.getElementById('weights-section');
    const svmSection = document.getElementById('svm-section');

    if (method === 'weighted_sum') {
        weightsSection.classList.remove('hidden');
        svmSection.classList.add('hidden');
    } else if (method === 'svm') {
        weightsSection.classList.add('hidden');
        svmSection.classList.remove('hidden');
    } else {
        weightsSection.classList.add('hidden');
        svmSection.classList.add('hidden');
    }
}

// Similarity Computation
async function computeSimilarityMatrix() {
    if (!state.embeddingsFolder) {
        alert('Please load an embeddings folder first');
        return;
    }

    const method = document.getElementById('similarity-method').value;
    const modelPath = method === 'svm' ? document.getElementById('svm-model-path').value.trim() : null;

    if (method === 'svm' && !modelPath) {
        alert('Please enter the SVM model path');
        return;
    }

    // Get transform threshold if enabled
    const transformEnabled = document.getElementById('transform-enabled').checked;
    const transformThreshold = transformEnabled ?
        parseFloat(document.getElementById('transform-threshold').value) : null;

    // Show spinner
    document.getElementById('compute-btn-text').textContent = 'Computing...';
    document.getElementById('compute-spinner').classList.remove('hidden');

    try {
        const response = await fetch('/api/compute-similarity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                embeddings_folder: state.embeddingsFolder,
                method: method,
                weights: method === 'weighted_sum' ? state.weights : null,
                weight_bounds: state.weightBounds,
                model_path: modelPath || document.getElementById('svm-model-path').value.trim(),
                transform_threshold: transformThreshold,
            }),
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Validate matrix is 2D
        if (!Array.isArray(data.matrix) || !Array.isArray(data.matrix[0])) {
            console.error('Matrix is not 2D:', data.matrix);
            alert('Error: Received invalid matrix format from server');
            return;
        }

        state.similarityMatrix = data.matrix;
        state.labels = data.labels;

        // Show sections
        document.getElementById('matrix-section').classList.remove('hidden');
        document.getElementById('save-matrix-btn').classList.remove('hidden');
        renderSideBySideMatrices();
    } catch (error) {
        alert(`Error computing similarity: ${error.message}`);
    } finally {
        document.getElementById('compute-btn-text').textContent = 'Compute Similarity Matrix';
        document.getElementById('compute-spinner').classList.add('hidden');
    }
}

function updateComparisonMatrix(threshold) {
    if (!state.similarityMatrix || !state.groundTruthMatrix) {
        console.warn("Cannot update comparison matrix: missing similarity or ground truth");
        state.comparisonMatrix = null;
        return;
    }

    const n = Math.min(state.similarityMatrix.length, state.groundTruthMatrix.length);
    console.log(`Computing comparison matrix for ${n}x${n} components at threshold ${threshold}`);

    state.comparisonMatrix = Array.from({ length: n }, () => new Array(n));

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const compVal = state.similarityMatrix[i][j];
            const gtVal = state.groundTruthMatrix[i][j];

            if (typeof compVal !== 'number' || typeof gtVal !== 'number' || isNaN(compVal) || isNaN(gtVal)) {
                state.comparisonMatrix[i][j] = NaN;
                continue;
            }

            const binarizedComp = compVal >= threshold ? 1 : 0;
            const binarizedGT = gtVal >= threshold ? 1 : 0;
            // Use specific values that we can detect in renderMatrixToHTML: 1 for Match, -1 for Mismatch
            state.comparisonMatrix[i][j] = (binarizedComp === binarizedGT) ? 1.0 : -1.0;
        }
    }
}

// Matrix Rendering
function renderSideBySideMatrices() {
    console.log("renderSideBySideMatrices called");
    const viewModeElement = document.getElementById('matrix-view-mode');
    const viewMode = viewModeElement ? viewModeElement.value : 'gt';
    const threshold = parseFloat(document.getElementById('gt-compare-threshold').value) || 0.5;

    console.log("View Mode:", viewMode);

    // Show/hide threshold container
    const thresholdContainer = document.getElementById('comparison-threshold-container');
    if (thresholdContainer) {
        thresholdContainer.classList.toggle('hidden', viewMode !== 'comparison');
    }

    // Render computed matrix on left
    const leftContainer = document.getElementById('matrix-container-left');
    if (state.similarityMatrix && state.labels) {
        leftContainer.innerHTML = renderMatrixToHTML(state.similarityMatrix, state.labels);
    } else {
        leftContainer.innerHTML = '<div class="text-gray-400 text-center py-8">No computed matrix</div>';
    }

    // Render right matrix based on mode
    const rightContainer = document.getElementById('matrix-container-right');
    const rightTitle = document.getElementById('right-matrix-title');

    if (viewMode === 'comparison') {
        if (state.similarityMatrix && state.groundTruthMatrix) {
            if (rightTitle) {
                rightTitle.textContent = 'COMPARISON VIEW: MATCH (Pink) vs MISMATCH (Grey/Blue)';
                // Use themed blue color instead of red
                rightTitle.style.color = state.isDarkMode ? 'var(--pastel-blue)' : '#2563EB';
                rightTitle.style.fontWeight = 'bold';
            }
            updateComparisonMatrix(threshold);
            if (state.comparisonMatrix) {
                rightContainer.innerHTML = renderMatrixToHTML(state.comparisonMatrix, state.labels, true);
            } else {
                rightContainer.innerHTML = '<div class="text-red-500 text-center py-8">Failed to compute comparison</div>';
            }
        } else {
            rightContainer.innerHTML = '<div class="text-gray-400 text-center py-8">Load Ground Truth to see comparison</div>';
        }
    } else {
        // Mode: Ground Truth
        if (rightTitle) {
            rightTitle.textContent = 'Ground Truth (Similarity Values)';
            rightTitle.style.color = 'green';
        }
        if (state.groundTruthMatrix) {
            const labels = state.labels.length > 0 ? state.labels :
                Array.from({ length: state.groundTruthMatrix.length }, (_, i) => `${i}`);
            rightContainer.innerHTML = renderMatrixToHTML(state.groundTruthMatrix, labels, false);
        } else {
            rightContainer.innerHTML = '<div class="text-gray-400 text-center py-8">No ground truth loaded</div>';
        }
    }
}

function renderMatrixToHTML(matrix, labels, isComparison = false, options = {}) {
    const n = matrix.length;
    const hideLabels = options.hideLabels || false;

    let html = '<table class="border-collapse" style="display: inline-table;">';

    // Header row
    if (!hideLabels) {
        html += '<tr><td class="matrix-cell matrix-header"></td>';
        for (let j = 0; j < n; j++) {
            const label = labels[j] || `${j}`;
            html += `<td class="matrix-cell matrix-header matrix-label-x" title="${label}">${label}</td>`;
        }
        html += '</tr>';
    }

    // Data rows
    for (let i = 0; i < n; i++) {
        const rowLabel = labels[i] || `${i}`;
        html += '<tr>';
        if (!hideLabels) {
            html += `<td class="matrix-cell matrix-header matrix-label-y" title="${rowLabel}">${rowLabel}</td>`;
        }

        // Ensure we have a valid row
        const row = matrix[i];
        if (!Array.isArray(row)) {
            // Single value or corrupted data - this is the bug!
            console.error('Invalid matrix row at index', i, '- expected array, got:', typeof row, row);
            html += `<td class="matrix-cell" style="background-color: #ccc" colspan="${n}">Invalid row data</td>`;
        } else {
            for (let j = 0; j < row.length; j++) {
                const value = row[j];
                if (typeof value !== 'number' || isNaN(value)) {
                    html += '<td class="matrix-cell" style="background-color: #ccc">?</td>';
                } else {
                    let color;
                    let displayValue;
                    if (isComparison) {
                        // Matching logic: 1.0 is Match, -1.0 is Mismatch
                        const isMatch = value > 0;
                        color = isMatch ? '#22c55e' : '#ef4444'; // Green vs Red
                        displayValue = isMatch ? '1.0' : '0.0';
                    } else {
                        color = getHeatmapColor(value);
                        displayValue = value.toFixed(2);
                    }

                    const colLabel = labels[j] || `${j}`;
                    const titleValue = value.toFixed(4);
                    // Use black text for better contrast on pastel backgrounds
                    html += `<td class="matrix-cell" style="background-color: ${color}; color: black;" title="${rowLabel} vs ${colLabel}: ${titleValue}">${displayValue}</td>`;
                }
            }
        }
        html += '</tr>';
    }

    html += '</table>';
    return html;
}

function getHeatmapColor(value) {
    // Interpolate between pastel colors based on value (0-1)
    const v = Math.max(0, Math.min(1, value));

    if (state.isDarkMode) {
        // Dark mode: desaturated pastels
        if (v < 0.5) {
            // Blue (#A8DADC) to Yellow (#FFE066)
            const t = v * 2;
            const r = Math.round(168 + (255 - 168) * t);
            const g = Math.round(218 + (224 - 218) * t);
            const b = Math.round(220 + (102 - 220) * t);
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // Yellow (#FFE066) to Pink (#FFC1CC)
            const t = (v - 0.5) * 2;
            const r = 255;
            const g = Math.round(224 + (193 - 224) * t);
            const b = Math.round(102 + (204 - 102) * t);
            return `rgb(${r}, ${g}, ${b})`;
        }
    } else {
        // Light mode: original pastel gradients
        if (v < 0.5) {
            // Blue to Yellow
            const t = v * 2;
            const r = Math.round(197 + (255 - 197) * t);
            const g = Math.round(232 + (243 - 232) * t);
            const b = Math.round(247 + (205 - 247) * t);
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // Yellow to Pink
            const t = (v - 0.5) * 2;
            const r = 255;
            const g = Math.round(243 + (214 - 243) * t);
            const b = Math.round(205 + (224 - 205) * t);
            return `rgb(${r}, ${g}, ${b})`;
        }
    }
}

async function saveMatrixAsImage(containerId, filename) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Temporarily disable scroll constraints for capture
    const originalMaxHeight = container.style.maxHeight;
    const originalOverflow = container.style.overflow;
    const originalWidth = container.style.width;
    const originalHeight = container.style.height;

    // Force container to expand to its full content size
    container.style.maxHeight = 'none';
    container.style.maxWidth = 'none';
    container.style.overflow = 'visible';
    container.style.width = 'max-content';
    container.style.height = 'auto';

    try {
        const canvas = await html2canvas(container, {
            backgroundColor: state.isDarkMode ? '#1a1a2e' : '#ffffff',
            logging: false,
            useCORS: true,
            width: container.scrollWidth,
            height: container.scrollHeight,
            windowWidth: container.scrollWidth + 100, // Ensure enough window width for capture
            windowHeight: container.scrollHeight + 100
        });

        const link = document.createElement('a');
        link.download = `${filename}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    } catch (error) {
        alert(`Error saving image: ${error.message}`);
    } finally {
        // Restore constraints
        container.style.maxHeight = originalMaxHeight;
        container.style.overflow = originalOverflow;
        container.style.width = originalWidth;
        container.style.height = originalHeight;
    }
}

// t-SNE Visualization with Plotly
function selectAllTsneEmbeddings() {
    for (let i = 0; i < 12; i++) {
        const checkbox = document.getElementById(`tsne-embed-${i}`);
        if (checkbox) checkbox.checked = true;
    }
}

function resetTsneScale() {
    document.getElementById('tsne-scale-factor').value = '1.0';
}

function onClusterMethodChange() {
    const method = document.getElementById('tsne-cluster-method').value;
    document.getElementById('kmeans-options').classList.toggle('hidden', method !== 'kmeans');
    document.getElementById('dbscan-options').classList.toggle('hidden', method !== 'dbscan');
}

async function computeTSNE() {
    if (!state.embeddingsFolder) {
        alert('Please load an embeddings folder first');
        return;
    }

    // Show spinner
    document.getElementById('tsne-btn-text').textContent = 'Computing...';
    document.getElementById('tsne-spinner').classList.remove('hidden');
    document.getElementById('tsne-progress').classList.remove('hidden');
    document.getElementById('tsne-status').textContent = 'Initializing t-SNE...';

    const perplexity = parseFloat(document.getElementById('tsne-perplexity').value) || 30;
    const scaleFactor = parseFloat(document.getElementById('tsne-scale-factor').value) || 1.0;

    // Collect selected embedding indices
    const selectedIndices = [];
    for (let i = 0; i < 12; i++) {
        const checkbox = document.getElementById(`tsne-embed-${i}`);
        if (checkbox && checkbox.checked) {
            selectedIndices.push(i);
        }
    }

    if (selectedIndices.length === 0) {
        alert('Please select at least one embedding');
        document.getElementById('tsne-btn-text').textContent = 'Compute t-SNE';
        document.getElementById('tsne-spinner').classList.add('hidden');
        return;
    }

    // Get threshold transform if enabled
    const transformEnabled = document.getElementById('tsne-transform-enabled')?.checked;
    const transformThreshold = transformEnabled ?
        parseFloat(document.getElementById('tsne-transform-threshold').value) : null;

    // Get clustering options
    const clusterMethod = document.getElementById('tsne-cluster-method')?.value || null;
    const clusterK = parseInt(document.getElementById('tsne-cluster-k')?.value) || 3;
    const clusterEps = parseFloat(document.getElementById('tsne-cluster-eps')?.value) || 0.5;
    const clusterMinSamples = parseInt(document.getElementById('tsne-cluster-min')?.value) || 2;

    try {
        const params = new URLSearchParams({
            embeddings_folder: state.embeddingsFolder,
            perplexity: perplexity.toString(),
            max_iter: '1000',
            scale_factor: scaleFactor.toString(),
            selected_indices: selectedIndices.join(','),
        });

        // Add optional params only if set
        if (transformThreshold !== null) {
            params.append('transform_threshold', transformThreshold.toString());
        }
        if (clusterMethod) {
            params.append('cluster_method', clusterMethod);
            if (clusterMethod === 'kmeans') {
                params.append('cluster_k', clusterK.toString());
            } else if (clusterMethod === 'dbscan') {
                params.append('cluster_eps', clusterEps.toString());
                params.append('cluster_min_samples', clusterMinSamples.toString());
            }
        }

        const response = await fetch(`/api/compute-tsne?${params}`, {
            method: 'POST',
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        state.tsneData = data;

        // Hide placeholder, show plot
        document.getElementById('tsne-placeholder').classList.add('hidden');
        document.getElementById('tsne-plot').classList.remove('hidden');

        renderTSNE3D();
    } catch (error) {
        alert(`Error computing t-SNE: ${error.message}`);
    } finally {
        document.getElementById('tsne-btn-text').textContent = 'Compute t-SNE';
        document.getElementById('tsne-spinner').classList.add('hidden');
        document.getElementById('tsne-progress').classList.add('hidden');
    }
}

function renderTSNE3D() {
    if (!state.tsneData) return;

    const { coordinates, labels, subfolder_indices, cluster_labels } = state.tsneData;

    // Determine coloring: by cluster if available, otherwise by subfolder
    let colors;
    let hoverLabels;

    if (cluster_labels && cluster_labels.length > 0) {
        // Color by cluster assignment using Tableau10
        colors = cluster_labels.map(cl => getTsnePointColor(cl, 10, true));
        hoverLabels = labels.map((label, i) => `${label} (Cluster ${cluster_labels[i]})`);
    } else {
        // Color by subfolder index using Tableau10
        const numSubfolders = Math.max(...subfolder_indices) + 1;
        colors = subfolder_indices.map(idx => getTsnePointColor(idx, numSubfolders));
        hoverLabels = labels;
    }

    const trace = {
        type: 'scatter3d',
        mode: 'markers',  // No text labels on points
        x: coordinates.map(c => c[0]),
        y: coordinates.map(c => c[1]),
        z: coordinates.map(c => c[2]),
        marker: {
            size: 8,
            color: colors,
            opacity: 0.9,
            line: { width: 0.5, color: 'rgba(0,0,0,0.3)' }
        },
        text: labels,  // Used for hover only
        hoverinfo: 'text',
        hovertext: hoverLabels,
    };

    const showGrid = document.getElementById('tsne-show-grid')?.checked !== false;

    const layout = {
        scene: {
            bgcolor: state.isDarkMode ? '#1a1a2e' : '#F8FAFC',
            xaxis: {
                showgrid: showGrid,
                gridcolor: showGrid ? (state.isDarkMode ? '#3d3d5c' : '#E5E7EB') : 'rgba(0,0,0,0)',
                zeroline: showGrid,
                zerolinecolor: showGrid ? (state.isDarkMode ? '#3d3d5c' : '#E5E7EB') : 'rgba(0,0,0,0)',
                showbackground: showGrid,
                backgroundcolor: state.isDarkMode ? '#1a1a2e' : '#F8FAFC',
                showticklabels: false,
                title: '',
            },
            yaxis: {
                showgrid: showGrid,
                gridcolor: showGrid ? (state.isDarkMode ? '#3d3d5c' : '#E5E7EB') : 'rgba(0,0,0,0)',
                zeroline: showGrid,
                zerolinecolor: showGrid ? (state.isDarkMode ? '#3d3d5c' : '#E5E7EB') : 'rgba(0,0,0,0)',
                showbackground: showGrid,
                backgroundcolor: state.isDarkMode ? '#1a1a2e' : '#F8FAFC',
                showticklabels: false,
                title: '',
            },
            zaxis: {
                showgrid: showGrid,
                gridcolor: showGrid ? (state.isDarkMode ? '#3d3d5c' : '#E5E7EB') : 'rgba(0,0,0,0)',
                zeroline: showGrid,
                zerolinecolor: showGrid ? (state.isDarkMode ? '#3d3d5c' : '#E5E7EB') : 'rgba(0,0,0,0)',
                showbackground: showGrid,
                backgroundcolor: state.isDarkMode ? '#1a1a2e' : '#F8FAFC',
                showticklabels: false,
                title: '',
            },
        },
        margin: { l: 0, r: 0, t: 0, b: 0 },
        hovermode: 'closest',
        paper_bgcolor: state.isDarkMode ? '#1a1a2e' : '#F8FAFC',
        font: { color: state.isDarkMode ? '#E4E4E4' : '#2D3436' },
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['toImage'],  // We have our own save
    };

    Plotly.newPlot('tsne-plot', [trace], layout, config);

    // Add click event
    const plotDiv = document.getElementById('tsne-plot');
    plotDiv.on('plotly_click', function (data) {
        if (data.points && data.points.length > 0) {
            // Use pointNumber as fallback for pointIndex
            const pointIndex = data.points[0].pointIndex !== undefined ?
                data.points[0].pointIndex : data.points[0].pointNumber;

            if (pointIndex !== undefined) {
                selectTsnePoint(pointIndex);
            }
        }
    });
}

// Tableau10 colormap - perceptually distinct, strong colors
// Source: D3.js schemeTableau10 (https://d3js.org/d3-scale-chromatic#schemeTableau10)
const TABLEAU10_COLORS = [
    '#4E79A7',  // Blue
    '#F28E2B',  // Orange
    '#E15759',  // Red
    '#76B7B2',  // Teal
    '#59A14F',  // Green
    '#EDC948',  // Yellow
    '#B07AA1',  // Purple
    '#FF9DA7',  // Pink
    '#9C755F',  // Brown
    '#BAB0AC',  // Gray
];

function getTsnePointColor(index, total, isCluster = false) {
    // For clusters, gray is used for noise (DBSCAN label -1)
    if (isCluster && index === -1) {
        return '#808080';  // Gray for noise points
    }
    return TABLEAU10_COLORS[index % TABLEAU10_COLORS.length];
}

function getPointColor(index, total) {
    // Pastel color palette (for non-t-SNE use, kept for compatibility)
    const colors = [
        '#FFD6E0', '#C5E8F7', '#B8E0D2', '#D4C4FB', '#FFEADD',
        '#FFF3CD', '#E8DAEF', '#D5F5E3', '#FADBD8', '#D6EAF8',
        '#F9E79F', '#ABEBC6'
    ];

    if (state.isDarkMode) {
        const darkColors = [
            '#FFC1CC', '#A8DADC', '#7FCDBB', '#B39CD0', '#FFDAC1',
            '#FFE066', '#C9A0DC', '#98D8C8', '#F5B7B1', '#85C1E9',
            '#F7DC6F', '#82E0AA'
        ];
        return darkColors[index % darkColors.length];
    }

    return colors[index % colors.length];
}

function selectTsnePoint(index) {
    if (!state.tsneData) return;

    const { labels, subfolder_indices, subfolders, cluster_labels } = state.tsneData;

    state.selectedPoint = index;
    const componentName = subfolders[subfolder_indices[index]];

    console.log('Selected point:', index, 'Component:', componentName, 'Label:', labels[index]);

    // Show selected info panel
    document.getElementById('tsne-selected-info').classList.remove('hidden');
    document.getElementById('selected-label').textContent = labels[index];
    document.getElementById('selected-subfolder').textContent = componentName;

    // Show cluster if available
    if (cluster_labels && cluster_labels.length > 0) {
        document.getElementById('selected-image-idx').textContent = `Cluster ${cluster_labels[index]}`;
    } else {
        document.getElementById('selected-image-idx').textContent = '-';
    }

    // Load full image list for the component to enable iteration
    loadTsneImages(componentName);
}

async function loadTsneImages(componentName) {
    if (!state.imagesFolder) return;

    try {
        const response = await fetch(`/api/images?folder=${encodeURIComponent(state.imagesFolder)}&subfolder=${encodeURIComponent(componentName)}`);
        const data = await response.json();

        if (data.images && data.images.length > 0) {
            state.tsneImages = data.images;
            state.tsneImageIndex = 0;

            // Show iteration controls
            document.getElementById('tsne-image-controls').classList.remove('hidden');
            updateTsneImageDisplay(componentName);
        } else {
            state.tsneImages = [];
            document.getElementById('tsne-image-controls').classList.add('hidden');
            document.getElementById('tsne-image-preview').innerHTML = '<span class="text-gray-400">No images found</span>';
        }
    } catch (error) {
        console.error('Error loading t-SNE images:', error);
    }
}

function nextTsneImage() {
    if (state.tsneImages.length === 0) return;
    state.tsneImageIndex = (state.tsneImageIndex + 1) % state.tsneImages.length;

    // We need to know which component it is. We can get it from state.tsneData
    const componentName = state.tsneData.subfolders[state.tsneData.subfolder_indices[state.selectedPoint]];
    updateTsneImageDisplay(componentName);
}

function prevTsneImage() {
    if (state.tsneImages.length === 0) return;
    state.tsneImageIndex = (state.tsneImageIndex - 1 + state.tsneImages.length) % state.tsneImages.length;

    const componentName = state.tsneData.subfolders[state.tsneData.subfolder_indices[state.selectedPoint]];
    updateTsneImageDisplay(componentName);
}

function updateTsneImageDisplay(componentName) {
    const filename = state.tsneImages[state.tsneImageIndex];
    const preview = document.getElementById('tsne-image-preview');
    const counter = document.getElementById('tsne-image-counter');
    const filenameDisplay = document.getElementById('selected-image-filename');

    const imgUrl = `/api/image?folder=${encodeURIComponent(state.imagesFolder)}&subfolder=${encodeURIComponent(componentName)}&filename=${encodeURIComponent(filename)}`;
    preview.innerHTML = `<img src="${imgUrl}" alt="${filename}">`;
    counter.textContent = `${state.tsneImageIndex + 1} / ${state.tsneImages.length}`;
    if (filenameDisplay) {
        filenameDisplay.textContent = filename;
    }
}

function deselectTsnePoint() {
    state.selectedPoint = null;
    state.tsneImages = [];
    document.getElementById('tsne-selected-info').classList.add('hidden');
    document.getElementById('tsne-image-controls').classList.add('hidden');
    const filenameDisplay = document.getElementById('selected-image-filename');
    if (filenameDisplay) filenameDisplay.textContent = '-';
    document.getElementById('tsne-image-preview').innerHTML =
        '<span class="text-gray-400">Select a point to view image</span>';
}

async function showSelectedImage(componentName) {
    const preview = document.getElementById('tsne-image-preview');

    if (!state.imagesFolder) {
        preview.innerHTML = '<span class="text-gray-400">Set images folder to view</span>';
        return;
    }

    console.log('Loading image for component:', componentName, 'from folder:', state.imagesFolder);

    try {
        // Use the component name as subfolder in the images folder
        const response = await fetch(`/api/images?folder=${encodeURIComponent(state.imagesFolder)}&subfolder=${encodeURIComponent(componentName)}`);
        const data = await response.json();

        console.log('Images API response:', data);

        if (data.error) {
            preview.innerHTML = `<span class="text-red-400">${data.error}</span>`;
            return;
        }

        if (data.images && data.images.length > 0) {
            const firstImage = data.images[0];
            const imgUrl = `/api/image?folder=${encodeURIComponent(state.imagesFolder)}&subfolder=${encodeURIComponent(componentName)}&filename=${encodeURIComponent(firstImage)}`;
            console.log('Loading image URL:', imgUrl);
            preview.innerHTML = `<img src="${imgUrl}" alt="${componentName}" onerror="this.parentElement.innerHTML='<span class=\\'text-red-400\\'>Image load failed</span>'">`;
        } else {
            preview.innerHTML = '<span class="text-gray-400">No images found</span>';
        }
    } catch (error) {
        console.error('Error loading image:', error);
        preview.innerHTML = `<span class="text-red-400">Error: ${error.message}</span>`;
    }
}

// Component Comparison
function onSubfolderChange(side) {
    const subfolder = document.getElementById(`${side}-subfolder`).value;
    if (!subfolder || !state.imagesFolder) return;

    loadImages(side, subfolder);
}

async function loadImages(side, subfolder) {
    try {
        const response = await fetch(`/api/images?folder=${encodeURIComponent(state.imagesFolder)}&subfolder=${encodeURIComponent(subfolder)}`);
        const data = await response.json();

        if (data.error) return;

        const select = document.getElementById(`${side}-image`);
        select.innerHTML = '<option value="">-- Select Image --</option>';

        data.images.forEach((img, idx) => {
            const option = document.createElement('option');
            option.value = img;
            // Use embedding label if available
            const label = state.embeddingLabels[idx] || img;
            option.textContent = `${idx}: ${label}`;
            select.appendChild(option);
        });

        // Memory selection logic
        if (state.lastImageIndex !== null && state.lastImageIndex < data.images.length) {
            select.selectedIndex = state.lastImageIndex + 1; // +1 for the placeholder
        } else if (data.images.length > 0) {
            select.selectedIndex = 1; // Select first image
        }
        onImageChange(side);
    } catch (error) {
        console.error('Error loading images:', error);
    }
}

function onImageChange(side) {
    const subfolder = document.getElementById(`${side}-subfolder`).value;
    const imageSelect = document.getElementById(`${side}-image`);
    const image = imageSelect.value;

    if (!image) return;

    // Save selection to memory
    state.lastImageIndex = imageSelect.selectedIndex - 1;

    // Coupled selection: sync the other side
    const otherSide = side === 'left' ? 'right' : 'left';
    const otherSelect = document.getElementById(`${otherSide}-image`);
    if (otherSelect && otherSelect.selectedIndex !== imageSelect.selectedIndex) {
        otherSelect.selectedIndex = imageSelect.selectedIndex;
        // Trigger update for other side's preview
        updateImagePreview(otherSide);
    }

    updateImagePreview(side);

    // Trigger comparison if both sides selected
    if (document.getElementById('left-subfolder').value && document.getElementById('right-subfolder').value) {
        compareSubfolders();
    }
}

function updateImagePreview(side) {
    const subfolder = document.getElementById(`${side}-subfolder`).value;
    const image = document.getElementById(`${side}-image`).value;
    if (!subfolder || !image) return;

    const imgUrl = `/api/image?folder=${encodeURIComponent(state.imagesFolder)}&subfolder=${encodeURIComponent(subfolder)}&filename=${encodeURIComponent(image)}`;
    document.getElementById(`${side}-image-preview`).innerHTML =
        `<img src="${imgUrl}" alt="${image}">`;
}

async function compareSubfolders() {
    const left = document.getElementById('left-subfolder').value;
    const right = document.getElementById('right-subfolder').value;

    if (!left || !right || !state.embeddingsFolder) return;

    const method = document.getElementById('similarity-method').value;

    // Get transform threshold if enabled
    const transformEnabled = document.getElementById('transform-enabled').checked;
    const transformThreshold = transformEnabled ?
        parseFloat(document.getElementById('transform-threshold').value) : null;

    try {
        const response = await fetch('/api/compare-subfolders', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                embeddings_folder: state.embeddingsFolder,
                left_subfolder: left,
                right_subfolder: right,
                method: method,
                weights: method === 'weighted_sum' ? state.weights : null,
                display_mode: 'corresponding',
                transform_threshold: transformThreshold,
                model_path: document.getElementById('svm-model-path').value.trim(),
            }),
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Store and display
        state.comparisonData = data;
        document.getElementById('comparison-section').classList.remove('hidden');
        updateComparison();
    } catch (error) {
        alert(`Error comparing components: ${error.message}`);
    }
}

function updateComparison() {
    if (!state.comparisonData) return;

    const { overall_similarity, individual_similarities } = state.comparisonData;

    // Update overall similarity
    document.getElementById('overall-similarity').textContent = overall_similarity.toFixed(4);

    // Update settings readout
    const method = document.getElementById('similarity-method').value;
    const transformEnabled = document.getElementById('transform-enabled').checked;
    const threshold = transformEnabled ? document.getElementById('transform-threshold').value : 'None';

    document.getElementById('comp-detail-method').textContent = `Method: ${method}`;
    document.getElementById('comp-detail-threshold').textContent = `Threshold: ${threshold}`;

    const weightsEl = document.getElementById('comp-detail-weights');
    if (method === 'weighted_sum') {
        weightsEl.classList.remove('hidden');
        const weightStr = state.weights.map(w => w.toFixed(2)).join(', ');
        weightsEl.textContent = `Weights: [${weightStr}]`;
    } else {
        weightsEl.classList.add('hidden');
    }

    // Always show corresponding bars now (pairwise matrix removed as per request)
    document.getElementById('similarity-details').classList.remove('hidden');
    document.getElementById('pairwise-matrix-container').classList.add('hidden');
    renderSimilarityBars(individual_similarities, method === 'svm');
}

function renderSimilarityBars(similarities, isGreyedOut = false) {
    const container = document.getElementById('similarity-bars');
    container.innerHTML = '';

    similarities.forEach((sim, i) => {
        const label = state.embeddingLabels[i] || `Pos ${i}`;
        const width = Math.max(0, Math.min(100, sim * 100));
        const opacityClass = isGreyedOut ? 'similarity-bar-greyed' : '';

        const div = document.createElement('div');
        div.className = `similarity-bar-container ${opacityClass}`;
        div.innerHTML = `
            <span class="similarity-bar-label" title="${label}">${label}</span>
            <div class="similarity-bar-track">
                <div class="similarity-bar-fill" style="width: ${width}%"></div>
            </div>
            <span class="similarity-bar-value">${sim.toFixed(3)}</span>
        `;
        container.appendChild(div);
    });
}

function renderPairwiseMatrix(matrix) {
    const container = document.getElementById('pairwise-matrix-container');
    const n = matrix.length;

    let html = '<table class="border-collapse ml-auto">';

    // Header
    html += '<tr><td class="matrix-cell matrix-header"></td>';
    for (let j = 0; j < n; j++) {
        const label = state.embeddingLabels[j] || `${j}`;
        html += `<td class="matrix-cell matrix-header matrix-label-x" title="${label}">${label}</td>`;
    }
    html += '</tr>';

    // Rows
    for (let i = 0; i < n; i++) {
        const rowLabel = state.embeddingLabels[i] || `${i}`;
        html += `<tr><td class="matrix-cell matrix-header matrix-label-y" title="${rowLabel}">${rowLabel}</td>`;
        for (let j = 0; j < n; j++) {
            const value = matrix[i][j];
            const color = getHeatmapColor(value);
            const colLabel = state.embeddingLabels[j] || `${j}`;
            html += `<td class="matrix-cell" style="background-color: ${color}" title="${rowLabel} vs ${colLabel}: ${value.toFixed(4)}">${value.toFixed(2)}</td>`;
        }
        html += '</tr>';
    }

    html += '</table>';
    container.innerHTML = html;
}

// Slot Management
function copyToSlot() {
    if (!state.comparisonData) {
        alert('Please perform a comparison first');
        return;
    }

    // Snapshot current state
    const slot = {
        id: Date.now(),
        left: document.getElementById('left-subfolder').value,
        right: document.getElementById('right-subfolder').value,
        data: JSON.parse(JSON.stringify(state.comparisonData)), // Deep copy data
        displayMode: document.getElementById('display-mode').value,
        embeddingLabels: [...state.embeddingLabels], // Copy labels
    };

    // Add to beginning, limit to 24
    state.slots.unshift(slot);
    if (state.slots.length > 24) {
        state.slots.pop();
    }

    renderSlots();

    // Scroll to the new slot
    setTimeout(() => {
        const slotsContainer = document.getElementById('comparison-slots');
        slotsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function deleteSlot(id) {
    state.slots = state.slots.filter(s => s.id !== id);
    renderSlots();
}

function renderSlots() {
    const container = document.getElementById('comparison-slots');
    container.innerHTML = '';

    if (state.slots.length === 0) {
        return;
    }

    state.slots.forEach((slot, index) => {
        const div = document.createElement('div');
        div.className = 'card slot-card relative';

        // Header with Delete button
        const header = `
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-md font-medium text-blue-600">Saved Comparison: ${slot.left} vs ${slot.right}</h3>
                <button onclick="deleteSlot(${slot.id})" class="text-red-500 hover:text-red-700 text-sm font-bold">
                    Delete Slot
                </button>
            </div>
        `;

        // Content (similar to comparison-section)
        const content = `
            <div class="grid grid-cols-2 gap-6">
                <div class="bg-gradient-to-r from-pastel-blue to-pastel-mint p-4 rounded-lg">
                    <div class="text-sm text-gray-600">Overall Similarity</div>
                    <div class="text-3xl font-light text-dark">${slot.data.overall_similarity.toFixed(4)}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    ${slot.displayMode === 'corresponding' ?
                `<div class="text-sm text-gray-600 mb-2">Individual Similarities</div>
                         <div id="slot-bars-${slot.id}"></div>` :
                `<div id="slot-matrix-${slot.id}" class="overflow-auto"></div>`
            }
                </div>
            </div>
        `;

        div.innerHTML = header + content;
        container.appendChild(div);

        // Render details after adding to DOM
        if (slot.displayMode === 'corresponding') {
            renderSlotBars(slot);
        } else {
            renderSlotMatrix(slot);
        }
    });
}

function renderSlotBars(slot) {
    const container = document.getElementById(`slot-bars-${slot.id}`);
    if (!container) return;

    slot.data.individual_similarities.forEach((sim, i) => {
        const label = slot.embeddingLabels[i] || `Pos ${i}`;
        const width = Math.max(0, Math.min(100, sim * 100));

        const barDiv = document.createElement('div');
        barDiv.className = 'similarity-bar-container';
        barDiv.innerHTML = `
            <span class="similarity-bar-label" title="${label}">${label}</span>
            <div class="similarity-bar-track">
                <div class="similarity-bar-fill" style="width: ${width}%"></div>
            </div>
            <span class="similarity-bar-value">${sim.toFixed(3)}</span>
        `;
        container.appendChild(barDiv);
    });
}

function renderSlotMatrix(slot) {
    const container = document.getElementById(`slot-matrix-${slot.id}`);
    if (!container) return;

    const matrix = slot.data.pairwise_matrix;
    const n = matrix.length;

    let html = '<table class="border-collapse ml-auto text-xs">';

    // Header
    html += '<tr><td class="matrix-cell matrix-header"></td>';
    for (let j = 0; j < n; j++) {
        const label = slot.embeddingLabels[j] || `${j}`;
        html += `<td class="matrix-cell matrix-header matrix-label-x" title="${label}">${label}</td>`;
    }
    html += '</tr>';

    // Rows
    for (let i = 0; i < n; i++) {
        const rowLabel = slot.embeddingLabels[i] || `${i}`;
        html += `<tr><td class="matrix-cell matrix-header matrix-label-y" title="${rowLabel}">${rowLabel}</td>`;
        for (let j = 0; j < n; j++) {
            const value = matrix[i][j];
            const color = getHeatmapColor(value);
            html += `<td class="matrix-cell p-1" style="background-color: ${color}">${value.toFixed(2)}</td>`;
        }
        html += '</tr>';
    }

    html += '</table>';
    container.innerHTML = html;
}

// Top-K Search
async function computeTopK() {
    const subfolder = document.getElementById('search-subfolder').value;
    const k = parseInt(document.getElementById('top-k-value').value) || 5;

    if (!subfolder || !state.embeddingsFolder) {
        alert('Please load embeddings and select a component first.');
        return;
    }

    const resultsContainer = document.getElementById('top-k-results');
    resultsContainer.innerHTML = '<div class="text-blue-500 text-center py-4">Searching...</div>';

    const method = document.getElementById('similarity-method').value;
    const transformEnabled = document.getElementById('transform-enabled').checked;
    const threshold = transformEnabled ? parseFloat(document.getElementById('transform-threshold').value) : null;
    const searchThreshold = parseFloat(document.getElementById('top-k-threshold').value) || 0.0;

    // Update settings readout
    const readout = document.getElementById('top-k-settings-readout');
    let settingsText = `Method: ${method}`;
    if (method === 'weighted_sum') {
        settingsText += ` (Weights: [${state.weights.map(w => w.toFixed(2)).join(', ')}])`;
    }
    if (threshold !== null) {
        settingsText += `, Threshold: ${threshold}`;
    }
    settingsText += `, Search Threshhold: ${searchThreshold}`;
    readout.textContent = `Search Settings: ${settingsText}`;

    try {
        const response = await fetch('/api/top-k', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                embeddings_folder: state.embeddingsFolder,
                component_name: subfolder,
                k: k,
                method: method,
                weights: method === 'weighted_sum' ? state.weights : null,
                similarity_threshold: searchThreshold,
                transform_threshold: threshold,
            }),
        });

        const data = await response.json();

        if (data.error) {
            resultsContainer.innerHTML = `<div class="text-red-400 text-center py-4">${data.error}</div>`;
            return;
        }

        renderTopKResults(data.matches);
    } catch (error) {
        resultsContainer.innerHTML = `<div class="text-red-400 text-center py-4">Error: ${error.message}</div>`;
    }
}

function renderTopKResults(matches) {
    const container = document.getElementById('top-k-results');
    container.innerHTML = '';

    if (!matches || matches.length === 0) {
        container.innerHTML = '<div class="text-gray-400 text-center py-4">No matches found</div>';
        return;
    }

    const itemBg = state.isDarkMode ? 'rgba(168, 218, 220, 0.1)' : 'rgba(197, 232, 247, 0.2)';
    const itemBorder = state.isDarkMode ? 'rgba(168, 218, 220, 0.3)' : 'var(--border-light)';

    matches.forEach((match, index) => {
        const div = document.createElement('div');
        div.className = 'flex items-center justify-between p-3 rounded-lg border mb-2 transition-all hover:translate-x-1';
        div.style.backgroundColor = itemBg;
        div.style.borderColor = itemBorder;

        // Thumbnail preview logic (if images folder is set)
        let thumbHtml = '';
        if (state.imagesFolder) {
            const imgUrl = `/api/image?folder=${encodeURIComponent(state.imagesFolder)}&subfolder=${encodeURIComponent(match.name)}&filename=0_image_0.png`;
            thumbHtml = `<img src="${imgUrl}" class="w-10 h-10 rounded object-cover shadow-sm mr-4" onerror="this.style.display='none'">`;
        }

        div.innerHTML = `
            <div class="flex items-center">
                <span class="w-6 text-gray-500 font-bold">${index + 1}</span>
                ${thumbHtml}
                <div>
                    <div class="font-medium text-dark">${match.name}</div>
                </div>
            </div>
            <div class="text-right">
                <div class="text-lg font-bold" style="color: ${state.isDarkMode ? 'var(--pastel-blue)' : '#2563EB'}" title="Raw Similarity: ${match.raw_similarity.toFixed(4)}">${match.score.toFixed(4)}</div>
                <div class="flex gap-2 mt-1">
                    <button onclick="selectTopKMatch('${match.name}', 'right')" class="btn-secondary text-[10px] px-2 py-0.5">to Right</button>
                </div>
            </div>
        `;
        container.appendChild(div);
    });
}

function selectTopKMatch(name, side) {
    const select = document.getElementById(`${side}-subfolder`);
    select.value = name;
    onSubfolderChange(side);
}

function onSearchSubfolderChange() {
    const val = document.getElementById('search-subfolder').value;
    if (val) {
        selectTopKMatch(val, 'left');
    }
}

function saveSimilarityMatrix() {
    if (!state.similarityMatrix || !state.labels) {
        alert('Please compute the similarity matrix first.');
        return;
    }

    const n = state.similarityMatrix.length;
    let csvContent = 'data:text/csv;charset=utf-8,';

    // Header (Labels)
    csvContent += ',' + state.labels.join(',') + '\n';

    // Rows
    for (let i = 0; i < n; i++) {
        csvContent += state.labels[i] + ',' + state.similarityMatrix[i].join(',') + '\n';
    }

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'similarity_matrix.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { state, getHeatmapColor, renderMatrixToHTML, reorderMatrix };
}


// Full Matrix Comparison View (Original Side-by-Side Pure View)
async function showFullMatrixComparison() {
    if (!state.similarityMatrix || !state.labels) {
        alert('Please compute or load a similarity matrix first');
        return;
    }

    const modal = document.getElementById('full-matrix-modal');
    const leftContainer = document.getElementById('modal-matrix-left');
    const rightContainer = document.getElementById('modal-matrix-right');

    leftContainer.innerHTML = '<div class="text-blue-500 py-20 text-center w-full">Generating pure matrix view...</div>';
    rightContainer.innerHTML = '';

    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    // Render HTML matrices directly
    leftContainer.innerHTML = renderPureMatrixToHTML(state.similarityMatrix, state.labels);
    if (state.groundTruthMatrix) {
        rightContainer.innerHTML = renderPureMatrixToHTML(state.groundTruthMatrix, state.labels);
    } else {
        rightContainer.innerHTML = '<div class="text-gray-400 text-center py-20">No ground truth loaded</div>';
    }
}

function renderPureMatrixToHTML(matrix, labels) {
    const n = matrix.length;
    if (n === 0) return '';
    let html = '<table class="border-collapse" style="display: inline-table;">';

    // Header row
    html += '<tr><td class="matrix-cell matrix-header"></td>';
    for (let j = 0; j < n; j++) {
        const label = labels[j] || `${j}`;
        html += `<td class="matrix-cell matrix-header matrix-label-x" title="${label}">${label}</td>`;
    }
    html += '</tr>';

    for (let i = 0; i < n; i++) {
        const rowLabel = labels[i] || `${i}`;
        html += `<tr><td class="matrix-cell matrix-header matrix-label-y" title="${rowLabel}">${rowLabel}</td>`;
        const row = matrix[i];
        for (let j = 0; j < row.length; j++) {
            const value = row[j];
            const color = getHeatmapColor(value);
            html += `<td class="matrix-cell" style="background-color: ${color}">${value.toFixed(2)}</td>`;
        }
        html += '</tr>';
    }

    html += '</table>';
    return html;
}

// Axis Reordering and GT Comparison
function applyAxisOrder() {
    const orderStr = document.getElementById('custom-axis-order').value.trim();
    if (!orderStr) return;

    const newLabels = orderStr.split(',').map(s => s.trim()).filter(s => s.length > 0);
    if (newLabels.length === 0) return;

    // Check if labels exist in current labels
    const missing = newLabels.filter(l => !state.labels.includes(l));
    if (missing.length > 0) {
        alert(`Labels not found: ${missing.join(', ')}`);
        return;
    }

    reorderMatrix(newLabels);
}

function reorderMatrix(newLabels) {
    if (!state.similarityMatrix) return;

    const oldLabels = [...state.labels];
    const n = oldLabels.length;

    // Create index mapping: newIndex -> oldIndex
    const mapping = newLabels.map(label => oldLabels.indexOf(label));

    // Reorder similarity matrix
    const newSimMatrix = [];
    for (let i = 0; i < mapping.length; i++) {
        const row = [];
        const oldRow = state.similarityMatrix[mapping[i]];
        for (let j = 0; j < mapping.length; j++) {
            row.push(oldRow[mapping[j]]);
        }
        newSimMatrix.push(row);
    }

    // Reorder ground truth matrix if it exists
    if (state.groundTruthMatrix && !state.preventGTReorder) {
        const newGTMatrix = [];
        for (let i = 0; i < mapping.length; i++) {
            const row = [];
            const oldRow = state.groundTruthMatrix[mapping[i]];
            for (let j = 0; j < mapping.length; j++) {
                row.push(oldRow[mapping[j]]);
            }
            newGTMatrix.push(row);
        }
        state.groundTruthMatrix = newGTMatrix;
    }

    state.similarityMatrix = newSimMatrix;
    state.labels = newLabels;

    // Update UI
    renderSideBySideMatrices();
    // Update dropdowns to match new order
    populateSubfolderDropdowns();
}

function openFullWidthView() {
    if (!state.similarityMatrix) {
        alert('Please compute similarity first');
        return;
    }

    const viewModeElement = document.getElementById('matrix-view-mode');
    const viewMode = viewModeElement ? viewModeElement.value : 'gt';
    const threshold = parseFloat(document.getElementById('gt-compare-threshold').value) || 0.5;

    const modal = document.getElementById('full-matrix-modal');
    const leftContainer = document.getElementById('modal-matrix-left');
    const rightContainer = document.getElementById('modal-matrix-right');
    const rightTitle = document.getElementById('modal-right-title');

    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    // Left: Computed similarity (no labels)
    leftContainer.innerHTML = renderMatrixToHTML(state.similarityMatrix, state.labels, false, { hideLabels: true });

    // Right: Selected mode (no labels)
    if (viewMode === 'comparison') {
        if (state.groundTruthMatrix) {
            updateComparisonMatrix(threshold);
            rightContainer.innerHTML = renderMatrixToHTML(state.comparisonMatrix, state.labels, true, { hideLabels: true });
        } else {
            rightContainer.innerHTML = '<div class="text-gray-400 text-center py-20">Ground Truth not loaded</div>';
        }
    } else {
        if (state.groundTruthMatrix) {
            rightContainer.innerHTML = renderMatrixToHTML(state.groundTruthMatrix, state.labels, false, { hideLabels: true });
        } else {
            rightContainer.innerHTML = '<div class="text-gray-400 text-center py-20">Ground Truth not loaded</div>';
        }
    }

    // Auto-scale matrices to fit view
    setTimeout(autoScaleModalMatrices, 100);
}

function autoScaleModalMatrices() {
    const containers = document.querySelectorAll('#full-matrix-modal .modal-matrix-container');
    containers.forEach((container, idx) => {
        const table = container.querySelector('table');
        if (!table) return;

        // Reset all styles that could affect measurement
        table.style.transform = 'none';
        table.style.marginLeft = '0';
        table.style.marginTop = '0';
        table.style.width = 'auto';
        table.style.height = 'auto';
        table.style.transformOrigin = 'top left';

        const containerRect = container.getBoundingClientRect();

        // Measure natural size (scrollWidth gets true content size even if container is smaller)
        const naturalWidth = table.scrollWidth;
        const naturalHeight = table.scrollHeight;

        const padding = 48; // Generous safety margin
        const availableWidth = containerRect.width - padding;
        const availableHeight = containerRect.height - padding;

        console.log(`Matrix ${idx} [${naturalWidth}x${naturalHeight}] into container [${availableWidth.toFixed(0)}x${availableHeight.toFixed(0)}]`);

        if (naturalWidth > 0 && naturalHeight > 0) {
            const scaleX = availableWidth / naturalWidth;
            const scaleY = availableHeight / naturalHeight;
            const scale = Math.min(scaleX, scaleY, 1.0); // Only scale down, never up

            console.log(`Applying scale: ${scale.toFixed(4)}`);

            table.style.transform = `scale(${scale})`;

            // Recalculate margins for centering based on EXACT scaled dimensions
            const scaledWidth = naturalWidth * scale;
            const scaledHeight = naturalHeight * scale;

            const marginLeft = Math.max(0, (availableWidth - scaledWidth) / 2);
            const marginTop = Math.max(0, (availableHeight - scaledHeight) / 2);

            table.style.marginLeft = `${marginLeft}px`;
            table.style.marginTop = `${marginTop}px`;
        }
    });
}

function closeFullMatrixModal() {
    document.getElementById('full-matrix-modal').classList.add('hidden');
    document.body.style.overflow = '';

    // Reset transforms and offsets
    const tables = document.querySelectorAll('#full-matrix-modal table');
    tables.forEach(table => {
        table.style.transform = 'none';
        table.style.marginLeft = '0';
        table.style.marginTop = '0';
    });
}

// Handle window resize for modal scaling
window.addEventListener('resize', () => {
    if (!document.getElementById('full-matrix-modal').classList.contains('hidden')) {
        autoScaleModalMatrices();
    }
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { state, getHeatmapColor, renderMatrixToHTML, reorderMatrix };
}

