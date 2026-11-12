// Evasive Maneuvers Visualizer - Frontend
let scene, camera, renderer, controls;
let cube, interceptorMarker, targetTrajectoryLine, targetPathSolid, interceptorPathLine, losLine;
let trajData = null;
let simData = null;
let currentFrame = 0;
let playing = false;
let animationId = null;

const CUBE_SIZE = 10000;

function initThree() {
    const container = document.getElementById('canvas-container');
    scene = new THREE.Scene();
    // Light theme like MATLAB
    scene.background = new THREE.Color(0xf8f9fa);
    
    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 1, 30000);
    // Better default view - isometric from above
    // Start with drone/interceptor view on ground level
    camera.position.set(500, 500, 2000);
    camera.lookAt(5000, 5000, 1000);
    
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(CUBE_SIZE/2, CUBE_SIZE/2, CUBE_SIZE/2);
    controls.update();
    
    // Create cube edges (clearer than wireframe fill)
    const cubeGeometry = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
    const edges = new THREE.EdgesGeometry(cubeGeometry);
    const cubeMaterial = new THREE.LineBasicMaterial({ 
        color: 0x1a1a1a, 
        linewidth: 2
    });
    cube = new THREE.LineSegments(edges, cubeMaterial);
    cube.position.set(CUBE_SIZE/2, CUBE_SIZE/2, CUBE_SIZE/2);
    scene.add(cube);
    
    // Add grid for better spatial reference (like MATLAB)
    const gridHelper = new THREE.GridHelper(CUBE_SIZE, 20, 0xcccccc, 0xe0e0e0);
    gridHelper.rotation.x = Math.PI / 2;
    gridHelper.position.set(CUBE_SIZE/2, CUBE_SIZE/2, 0);
    scene.add(gridHelper);
    
    // Add axes with clearer colors
    const axesHelper = new THREE.AxesHelper(2000);
    axesHelper.position.set(0, 0, 0);
    scene.add(axesHelper);
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.3);
    directionalLight.position.set(10000, 10000, 10000);
    scene.add(directionalLight);
    
    // Interceptor marker - bright green, larger
    const intGeometry = new THREE.SphereGeometry(80, 16, 16);
    const intMaterial = new THREE.MeshBasicMaterial({ color: 0x00aa00 });
    interceptorMarker = new THREE.Mesh(intGeometry, intMaterial);
    scene.add(interceptorMarker);
    
    animate();
}

function animate() {
    animationId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
    
    if (playing && simData) {
        updatePlayback();
    }
}

function updatePlayback() {
    if (simData && Array.isArray(simData.traj_T) && Array.isArray(simData.traj_M)) {
        if (currentFrame < simData.traj_T.length && currentFrame < simData.traj_M.length) {
            // Update LOS
            if (losLine) {
                scene.remove(losLine);
            }
            const tPos = simData.traj_T[currentFrame];
            const mPos = simData.traj_M[currentFrame];
            const losGeometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(mPos[0], mPos[1], mPos[2]),
                new THREE.Vector3(tPos[0], tPos[1], tPos[2])
            ]);
            const losMaterial = new THREE.LineBasicMaterial({ color: 0xff6600 });
            losLine = new THREE.Line(losGeometry, losMaterial);
            scene.add(losLine);
            
            // Update interceptor marker position
            if (interceptorMarker) {
                interceptorMarker.position.set(mPos[0], mPos[1], mPos[2]);
            }
            // Update target marker position
            if (window.targetMarker) {
                window.targetMarker.position.set(tPos[0], tPos[1], tPos[2]);
            }
            
            // Update data
            updateDataDisplay(currentFrame);
            currentFrame++;
        } else {
            playing = false;
        }
    }
}

function updateDataDisplay(frame) {
    if (!simData) return;
    if (!simData.traj_T || !simData.traj_M) return;
    const tPos = simData.traj_T[frame] || simData.traj_T[simData.traj_T.length-1];
    const mPos = simData.traj_M[frame] || simData.traj_M[simData.traj_M.length-1];
    const content = document.getElementById('data-content');
    content.innerHTML = `
        <div>Frame: ${frame} / ${simData.traj_T.length}</div>
        <div>Target Pos: [${tPos[0].toFixed(1)}, ${tPos[1].toFixed(1)}, ${tPos[2].toFixed(1)}]</div>
        <div>Interceptor Pos: [${mPos[0].toFixed(1)}, ${mPos[1].toFixed(1)}, ${mPos[2].toFixed(1)}]</div>
        <div>Target Vel: [${(simData.final_v_T[0]||0).toFixed(1)}, ${(simData.final_v_T[1]||0).toFixed(1)}, ${(simData.final_v_T[2]||0).toFixed(1)}]</div>
        <div>Target Accel: [${(simData.final_a_T[0]||0).toFixed(1)}, ${(simData.final_a_T[1]||0).toFixed(1)}, ${(simData.final_a_T[2]||0).toFixed(1)}]</div>
        <div>Miss Distance: ${(simData.miss_distance||0).toFixed(2)} m</div>
    `;
}

async function loadManeuvers() {
    const response = await fetch('/api/maneuvers');
    const maneuvers = await response.json();
    const select = document.getElementById('maneuver-select');
    select.innerHTML = '';
    // Add No Evasion option first
    const noneOpt = document.createElement('option');
    noneOpt.value = 'none';
    noneOpt.textContent = 'NO EVASION';
    select.appendChild(noneOpt);
    maneuvers.forEach(m => {
        if (m.id === 'none') return;
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name.replace(/_/g, ' ');
        select.appendChild(opt);
    });
}

let trajType = 'polynomial';
let orderCount = 2;

function setupControls() {
    // Interceptor position sliders
    document.getElementById('int-x').addEventListener('input', (e) => {
        document.getElementById('int-x-val').textContent = e.target.value;
        updateInterceptorMarker();
    });
    document.getElementById('int-y').addEventListener('input', (e) => {
        document.getElementById('int-y-val').textContent = e.target.value;
        updateInterceptorMarker();
    });
    
    // Trajectory type switch
    document.getElementById('traj-switch').addEventListener('click', () => {
        const btn = document.getElementById('traj-switch');
        btn.classList.toggle('active');
        if (btn.classList.contains('active')) {
            trajType = 'bezier';
            document.getElementById('order-label').textContent = 'Point Count';
            document.getElementById('eta-group').style.display = 'block';
            if (orderCount < 2) orderCount = 2;
            if (orderCount > 8) orderCount = 8;
        } else {
            trajType = 'polynomial';
            document.getElementById('order-label').textContent = 'Order';
            document.getElementById('eta-group').style.display = 'none';
            if (orderCount < 1) orderCount = 1;
            if (orderCount > 4) orderCount = 4;
        }
        document.getElementById('order-value').textContent = orderCount;
    });
    
    // Order controls
    document.getElementById('order-dec').addEventListener('click', () => {
        if (trajType === 'polynomial') {
            if (orderCount > 1) orderCount--;
        } else {
            if (orderCount > 2) orderCount--;
        }
        document.getElementById('order-value').textContent = orderCount;
    });
    document.getElementById('order-inc').addEventListener('click', () => {
        if (trajType === 'polynomial') {
            if (orderCount < 4) orderCount++;
        } else {
            if (orderCount < 8) orderCount++;
        }
        document.getElementById('order-value').textContent = orderCount;
    });
    
    // Eta slider
    document.getElementById('eta').addEventListener('input', (e) => {
        document.getElementById('eta-val').textContent = e.target.value;
    });
    
    // Refresh trajectory
    document.getElementById('refresh-traj').addEventListener('click', generateTrajectory);
    
    // Spec sliders
    document.getElementById('max-vel').addEventListener('input', (e) => {
        document.getElementById('vel-val').textContent = e.target.value;
    });
    document.getElementById('max-accel').addEventListener('input', (e) => {
        document.getElementById('accel-val').textContent = e.target.value;
    });
    document.getElementById('nav-ratio').addEventListener('input', (e) => {
        document.getElementById('nav-val').textContent = e.target.value;
    });
    document.getElementById('max-tvel').addEventListener('input', (e) => {
        document.getElementById('tvel-val').textContent = e.target.value;
    });
    document.getElementById('max-taccel').addEventListener('input', (e) => {
        document.getElementById('taccel-val').textContent = e.target.value;
    });
    document.getElementById('max-drone').addEventListener('input', (e) => {
        document.getElementById('drone-val').textContent = e.target.value;
    });
    
    // Tabs
    document.getElementById('tab-interceptor').addEventListener('click', () => {
        document.getElementById('interceptor-specs').style.display = 'block';
        document.getElementById('target-specs').style.display = 'none';
        document.getElementById('tab-interceptor').style.background = '#4a9eff';
        document.getElementById('tab-target').style.background = '#333';
    });
    document.getElementById('tab-target').addEventListener('click', () => {
        document.getElementById('interceptor-specs').style.display = 'none';
        document.getElementById('target-specs').style.display = 'block';
        document.getElementById('tab-target').style.background = '#4a9eff';
        document.getElementById('tab-interceptor').style.background = '#333';
    });
    
    // Simulate
    document.getElementById('simulate-btn').addEventListener('click', runSimulation);
    
    // Playback
    document.getElementById('play-btn').addEventListener('click', () => {
        playing = true;
        if (currentFrame >= (simData ? simData.traj_T.length : 0)) {
            currentFrame = 0;
        }
    });
    document.getElementById('pause-btn').addEventListener('click', () => {
        playing = false;
    });
    document.getElementById('reset-btn').addEventListener('click', () => {
        playing = false;
        currentFrame = 0;
        if (losLine) {
            scene.remove(losLine);
            losLine = null;
        }
        if (simData && simData.traj_M && simData.traj_M.length > 0) {
            interceptorMarker.position.set(simData.traj_M[0][0], simData.traj_M[0][1], simData.traj_M[0][2]);
        } else {
            updateInterceptorMarker();
        }
        if (simData && simData.traj_T && simData.traj_T.length > 0 && window.targetMarker) {
            window.targetMarker.position.set(simData.traj_T[0][0], simData.traj_T[0][1], simData.traj_T[0][2]);
            updateDataDisplay(0);
        }
    });
}

function updateInterceptorMarker() {
    const x = parseFloat(document.getElementById('int-x').value);
    const y = parseFloat(document.getElementById('int-y').value);
    if (interceptorMarker) {
        interceptorMarker.position.set(x, y, 50); // On ground level
    }
}

async function generateTrajectory() {
    const eta = parseFloat(document.getElementById('eta').value);
    const response = await fetch('/api/generate_trajectory', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            type: trajType,
            order: orderCount,
            point_count: orderCount,
            eta: eta
        })
    });
    trajData = await response.json();
    displayTrajectory();
}

function displayTrajectory() {
    if (targetTrajectoryLine) {
        scene.remove(targetTrajectoryLine);
    }
    if (!trajData || !trajData.points) return;
    
    const points = trajData.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineDashedMaterial({ 
        color: 0xff0000, 
        dashSize: 150, 
        gapSize: 80,
        linewidth: 2
    });
    targetTrajectoryLine = new THREE.Line(geometry, material);
    targetTrajectoryLine.computeLineDistances();
    scene.add(targetTrajectoryLine);
}

async function runSimulation() {
    if (!trajData) {
        alert('Generate a trajectory first!');
        return;
    }
    const x = parseFloat(document.getElementById('int-x').value);
    const y = parseFloat(document.getElementById('int-y').value);
    const maneuver = document.getElementById('maneuver-select').value;
    const maxVel = parseFloat(document.getElementById('max-vel').value);
    const maxAccel = parseFloat(document.getElementById('max-accel').value);
    const maxDrone = parseFloat(document.getElementById('max-drone').value);
    const navRatio = parseFloat(document.getElementById('nav-ratio').value);
    const targetPlatform = document.getElementById('target-platform').value;
    const maxTVel = parseFloat(document.getElementById('max-tvel').value);
    const maxTAccel = parseFloat(document.getElementById('max-taccel').value);
    
    const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            trajectory: trajData.points,
            interceptor_pos: [x, y],
            maneuver: maneuver,
            max_vel: maxVel,
            max_accel: maxAccel,
            max_drone_speed: maxDrone,
            nav_ratio: navRatio,
            target_platform: targetPlatform,
            max_tvel: maxTVel,
            max_taccel: maxTAccel
        })
    });
    simData = await response.json();
    
    if (simData.error) {
        alert('Simulation error: ' + simData.error);
        return;
    }
    
    document.getElementById('miss-dist').textContent = `Miss Distance: ${simData.miss_distance.toFixed(2)} m`;
    document.getElementById('playback-controls').style.display = 'flex';
    currentFrame = 0;
    playing = false;
    
    // Remove old target marker if exists
    if (window.targetMarker) {
        scene.remove(window.targetMarker);
    }
    
    // Display solid target path
    if (targetPathSolid) scene.remove(targetPathSolid);
    if (simData.traj_T && simData.traj_T.length > 0) {
        const solidPoints = simData.traj_T.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        const solidGeom = new THREE.BufferGeometry().setFromPoints(solidPoints);
        const solidMat = new THREE.LineBasicMaterial({ color: 0xcc0000, linewidth: 3 });
        targetPathSolid = new THREE.Line(solidGeom, solidMat);
        scene.add(targetPathSolid);
        
        // Add target marker (red sphere)
        const tGeom = new THREE.SphereGeometry(60, 16, 16);
        const tMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        window.targetMarker = new THREE.Mesh(tGeom, tMat);
        window.targetMarker.position.set(simData.traj_T[0][0], simData.traj_T[0][1], simData.traj_T[0][2]);
        scene.add(window.targetMarker);
    }
    
    // Display interceptor path
    if (interceptorPathLine) scene.remove(interceptorPathLine);
    if (simData.traj_M && simData.traj_M.length > 0) {
        const intPoints = simData.traj_M.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        const intGeom = new THREE.BufferGeometry().setFromPoints(intPoints);
        const intMat = new THREE.LineBasicMaterial({ color: 0x0066ff, linewidth: 2 });
        interceptorPathLine = new THREE.Line(intGeom, intMat);
        scene.add(interceptorPathLine);
        // Set marker to start position
        if (interceptorMarker && simData.traj_M[0]) {
            interceptorMarker.position.set(simData.traj_M[0][0], simData.traj_M[0][1], simData.traj_M[0][2]);
        }
        if (window.targetMarker && simData.traj_T[0]) {
            window.targetMarker.position.set(simData.traj_T[0][0], simData.traj_T[0][1], simData.traj_T[0][2]);
        }
    }
}

function onWindowResize() {
    const container = document.getElementById('canvas-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

window.addEventListener('resize', onWindowResize);

window.onload = () => {
    initThree();
    loadManeuvers();
    setupControls();
    updateInterceptorMarker();
};
