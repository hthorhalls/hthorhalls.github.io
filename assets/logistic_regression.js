// Get the div where we want to attach the plot
const container = document.getElementById('logistic_plot');

// Set up the Three.js scene, camera, and renderer
const scene = new THREE.Scene();

// Enable antialiasing for smoother rendering
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setClearColor(0x000000, 0);  // 0 for transparent background
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

// Camera setup
const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
camera.position.set(6, 6, 6); // Adjust camera position

// OrbitControls (to allow rotating, zooming, and panning)
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; // Smooth motion
controls.dampingFactor = 0.05;

// Lighting setup
const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(0, 0, 10).normalize();
scene.add(light);

// Helper function to generate random normal (Gaussian) values
function randn_bm() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Generate multivariate Gaussian data
function generateGaussianData(mean, cov, n) {
    const data = [];
    for (let i = 0; i < n; i++) {
        const x = randn_bm() * cov[0] + mean[0];
        const y = randn_bm() * cov[1] + mean[1];
        const z = 0;
        data.push([x, y, z]);
    }
    return data;
}

const mean1 = [0.2, 0.3, 0.1];
const cov1 = [0.1, 0.1, 0.1];

const mean2 = [-0.2, -0.3, 0.6];
const cov2 = [0.1, 0.1, 0.1];

const class1Data = generateGaussianData(mean1, cov1, 100);
const class2Data = generateGaussianData(mean2, cov2, 100);

// Function to add data points to the scene using Points
function addDataPointsAsPoints(data, color) {
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array(data.length * 3);

    for (let i = 0; i < data.length; i++) {
        vertices[i * 3] = data[i][0];
        vertices[i * 3 + 1] = data[i][1];
        vertices[i * 3 + 2] = data[i][2];
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

    const material = new THREE.PointsMaterial({ color, size: 0.1 });
    const points = new THREE.Points(geometry, material);
    scene.add(points);
}

// Use this function to add data points
addDataPointsAsPoints(class1Data, 0x0000ff);  // Blue
addDataPointsAsPoints(class2Data, 0xff0000);  // Red

// Add coordinate axes (x, y, z) with labels
const axesLength = 1;
const axisMaterial = new THREE.LineBasicMaterial({ color: 0x000000 }); // Black color

// X Axis
const xGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(axesLength, 0, 0)
]);
const xAxis = new THREE.Line(xGeometry, axisMaterial);
scene.add(xAxis);

// Y Axis
const yGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, axesLength, 0)
]);
const yAxis = new THREE.Line(yGeometry, axisMaterial);
scene.add(yAxis);

// Z Axis
const zGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, axesLength)
]);
const zAxis = new THREE.Line(zGeometry, axisMaterial);
scene.add(zAxis);

// Create labels for axes
const fontLoader = new THREE.FontLoader();
fontLoader.load('https://unpkg.com/three@0.77.0/examples/fonts/helvetiker_regular.typeface.json', (font) => {
    const createLabel = (text, position) => {
        const geometry = new THREE.TextGeometry(text, {
            font: font,
            size: 0.5,
            height: 0.1
        });
        const material = new THREE.MeshBasicMaterial({ color: 0x000000 }); // Black color
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(position.x, position.y, position.z);
        mesh.geometry.computeBoundingBox();  // Ensure bounding box is computed
        const bbox = mesh.geometry.boundingBox;
        mesh.position.sub(bbox.getCenter(new THREE.Vector3())); // Center the text
        scene.add(mesh);
    };

    createLabel('X', { x: axesLength + 0.5, y: 0, z: 0 });
    createLabel('Y', { x: 0, y: axesLength + 0.5, z: 0 });
    createLabel('Z', { x: 0, y: 0, z: axesLength + 0.5 });
});

// Add a 3D grid
const gridHelper = new THREE.GridHelper(10, 10, 0x808080, 0x808080); // Grey color
gridHelper.rotation.x = Math.PI / 2;  // Rotate grid to be horizontal
gridHelper.position.y = -0.01;  // Slightly below the data points to avoid overlap
scene.add(gridHelper);

// Function to compute logistic function
function logisticFunction(x, y) {
    const z = x * 1 + y * 1 + (-0.5);  // Example parameters
    return 1 / (1 + Math.exp(-z));
}

// Function to add logistic function as a surface with gradient color
function addLogisticSurface(xMin, xMax, yMin, yMax, step) {
    const geometry = new THREE.Geometry();
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    for (let x = xMin; x <= xMax; x += step) {
        for (let y = yMin; y <= yMax; y += step) {
            const z = logisticFunction(x, y);
            geometry.vertices.push(new THREE.Vector3(x, y, z));
        }
    }

    // Create faces for the surface
    const numPointsX = (xMax - xMin) / step + 1;
    for (let i = 0; i < numPointsX - 1; i++) {
        for (let j = 0; j < numPointsX - 1; j++) {
            const a = i * numPointsX + j;
            const b = (i + 1) * numPointsX + j;
            const c = (i + 1) * numPointsX + (j + 1);
            const d = i * numPointsX + (j + 1);
            geometry.faces.push(new THREE.Face3(a, b, d));
            geometry.faces.push(new THREE.Face3(b, c, d));
        }
    }

    geometry.computeFaceNormals();
    geometry.computeVertexNormals();

    // Vertex shader for gradient color
    const vertexShader = `
        varying float vZ;
        void main() {
            vZ = position.z;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `;

    // Fragment shader for gradient color
    const fragmentShader = `
        varying float vZ;
        void main() {
            vec3 color1 = vec3(0.0, 0.0, 1.0); // Blue
            vec3 color2 = vec3(1.0, 0.0, 0.0); // Red
            float t = (vZ - 0.0) / 1.0; // Adjust normalization if needed
            vec3 color = mix(color1, color2, t);
            gl_FragColor = vec4(color, 1.0);
        }
    `;

    const material = new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        side: THREE.DoubleSide,
        transparent: true
    });

    const surface = new THREE.Mesh(geometry, material);
    scene.add(surface);
}

// Add the logistic surface with gradient
addLogisticSurface(-1, 1, -1, 1, 0.5);

// Render loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();  // Update controls to allow interaction
    renderer.render(scene, camera);
}

animate();

// Resize event listener to keep canvas responsive
window.addEventListener('resize', function () {
    const width = container.clientWidth;
    const height = container.clientHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
});
