// ======================
// 🌍 GLOBALS
// ======================
let map = null;
let routeLayers = [];
let selectedRoute = null;

let userMarker = null;
let watchId = null;
let routeGenerated = false;

let destinationPoint = null;

let startPoint = null;
let goalPoint = null;

let lastPosition = null;
// ======================
// 🚨 NEW: RISK ZONES
// ======================
let riskZones = [
    {x: 220, y: 180, radius: 40, level: "high"},
    {x: 300, y: 260, radius: 30, level: "medium"}
];

let riskLayers = [];


// ======================
// 🔁 NEW: PIXEL → GEO
// ======================
function pixelToLatLon(x, y){

    const topLat = parseFloat(localStorage.getItem("top_lat"));
    const leftLon = parseFloat(localStorage.getItem("left_lon"));
    const bottomLat = parseFloat(localStorage.getItem("bottom_lat"));
    const rightLon = parseFloat(localStorage.getItem("right_lon"));

    if(!topLat || !leftLon || !bottomLat || !rightLon) return null;

    const lat = topLat + (y / 512) * (bottomLat - topLat);
    const lon = leftLon + (x / 512) * (rightLon - leftLon);

    return [lat, lon];
}


// ======================
// 🔥 ARROW ICON
// ======================
function getArrowIcon(angle){
    return L.divIcon({
        className: "arrow-icon",
        html: `<div style="
            transform: rotate(${angle}deg);
            font-size: 22px;
            color:#00ff88;
        ">▲</div>`,
        iconSize: [20,20]
    });
}


// ======================
// 🌍 DRAW MAP ROUTES
// ======================
function drawMapRoutes(routes){

    if(!routes || routes.length === 0) return;

    const mapDiv = document.getElementById("map");
    mapDiv.style.display = "block";

    if(!map){
        map = L.map('map').setView(routes[0].geo_path[0], 15);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png')
        .addTo(map);
    }

    // clear old
    routeLayers.forEach(l => map.removeLayer(l));
    routeLayers = [];

    const colors = {
        "Best": "#00ff88",
        "Safe": "#00ccff",
        "Alternative": "#ffcc00"
    };

    routes.forEach((route) => {

        const line = L.polyline(route.geo_path, {
            color: colors[route.label] || "#888",
            weight: 10,
            opacity: 0.9
        }).addTo(map);

        line.on("click", function(e){

            selectedRoute = route.geo_path;
            destinationPoint = selectedRoute[selectedRoute.length - 1];

            routeLayers.forEach(l => l.setStyle({weight:6, opacity:0.4}));

            line.setStyle({weight:14, opacity:1});

            alert("🧭 Selected: " + route.label);

            L.DomEvent.stopPropagation(e);
        });

        routeLayers.push(line);
    });

    map.fitBounds(routeLayers[0].getBounds());

    setTimeout(() => map.invalidateSize(), 300);
}


// ======================
// 🚨 NEW: DRAW RISK ON MAP
// ======================
function drawRiskZonesOnMap(){

    if(!map) return;

    riskZones.forEach(zone => {

        const geo = pixelToLatLon(zone.x, zone.y);
        if(!geo) return;

        const color = zone.level === "high" ? "red" : "orange";

        const circle = L.circle(geo, {
            color: color,
            fillColor: color,
            fillOpacity: 0.35,
            radius: zone.radius * 2
        }).addTo(map);

        riskLayers.push(circle);
    });
}


// ======================
// 🖼 DRAW IMAGE
// ======================
function drawImage(){

    const canvas = document.getElementById("resultCanvas");
    const ctx = canvas.getContext("2d");
    const img = canvas._img;

    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);

    const drawW = img.width * scale;
    const drawH = img.height * scale;

    const offsetX = (canvas.width - drawW) / 2;
    const offsetY = (canvas.height - drawH) / 2;

    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(img, offsetX, offsetY, drawW, drawH);

    canvas._scale = scale;
    canvas._offsetX = offsetX;
    canvas._offsetY = offsetY;
}


// ======================
// 🚨 NEW: DRAW RISK ON IMAGE
// ======================
function drawRiskZones(){

    const canvas = document.getElementById("resultCanvas");
    const ctx = canvas.getContext("2d");

    riskZones.forEach(zone => {

        const x = zone.x * canvas._scale + canvas._offsetX;
        const y = zone.y * canvas._scale + canvas._offsetY;
        const r = zone.radius * canvas._scale;

        ctx.fillStyle = zone.level === "high"
            ? "rgba(255,0,0,0.35)"
            : "rgba(255,165,0,0.35)";

        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2*Math.PI);
        ctx.fill();
    });
}



// ======================
// 🎯 DRAW POINT
// ======================
function drawPoint(p, color){

    const canvas = document.getElementById("resultCanvas");
    const ctx = canvas.getContext("2d");

    const x = p.x * canvas._scale + canvas._offsetX;
    const y = p.y * canvas._scale + canvas._offsetY;

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2*Math.PI);
    ctx.fill();
}


// ======================
// 🔥 SMOOTH PATH
// ======================
function smoothPath(path){

    const smooth = [];

    for(let i = 0; i < path.length - 1; i++){
        const p1 = path[i];
        const p2 = path[i+1];

        smooth.push(p1);
        smooth.push([
            (p1[0] + p2[0]) / 2,
            (p1[1] + p2[1]) / 2
        ]);
    }

    smooth.push(path[path.length - 1]);

    return smooth;
}


// ======================
// 🖼 DRAW ROUTES
// ======================
function drawAllRoutes(routes){

    const canvas = document.getElementById("resultCanvas");
    const ctx = canvas.getContext("2d");

    drawImage();

    const img = canvas._img;

    const colors = {
        "Best": "#00ff88",
        "Safe": "#00ccff",
        "Alternative": "#ffcc00"
    };

    routes.forEach(route => {

        if(!route.path) return;

        const path = smoothPath(route.path);

        ctx.beginPath();
        ctx.strokeStyle = colors[route.label] || "#888";
        ctx.lineWidth = route.label === "Best" ? 6 : 3;

        path.forEach((p, i) => {

            const px = p[1] * (img.width / 512);
            const py = p[0] * (img.height / 512);

            const x = px * canvas._scale + canvas._offsetX;
            const y = py * canvas._scale + canvas._offsetY;

            if(i === 0) ctx.moveTo(x,y);
            else ctx.lineTo(x,y);
        });

        ctx.stroke();
    });

    if(startPoint) drawPoint(startPoint, "#00ff88");
    if(goalPoint) drawPoint(goalPoint, "#ff3b3b");
}


// ======================
// 📊 LEGEND
// ======================
function updateLegend(routes){

    const legend = document.getElementById("routeLegend");

    const colors = {
        "Best": "#00ff88",
        "Safe": "#00ccff",
        "Alternative": "#ffcc00"
    };

    legend.innerHTML = "<b>Routes:</b><br>";

    routes.forEach(r => {
        legend.innerHTML += `
            <div style="margin:5px;">
                <span style="background:${colors[r.label]};
                width:12px;height:12px;display:inline-block;border-radius:50%;margin-right:6px;"></span>
                ${r.label} (${r.distance_km ?? "--"} km)
            </div>
        `;
    });
}


// ======================
// 🚀 NAVIGATION
// ======================
function startNavigation(){

    if(!selectedRoute){
        alert("Select route on map first");
        return;
    }

    watchId = navigator.geolocation.watchPosition(pos => {

        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;

        const current = [lat, lon];

        let angle = 0;

        if(lastPosition){
            const dx = lon - lastPosition[1];
            const dy = lat - lastPosition[0];
            angle = Math.atan2(dx, dy) * (180 / Math.PI);
        }

        lastPosition = current;

        if(!userMarker){
            userMarker = L.marker(current, {
                icon: getArrowIcon(angle)
            }).addTo(map);
        } else {
            userMarker.setLatLng(current);
            userMarker.setIcon(getArrowIcon(angle));
        }

        map.flyTo(current, map.getZoom(), {duration:0.5});

        checkOffRoute(current);

    }, err => {
        alert("GPS Error: " + err.message);
    }, {
        enableHighAccuracy: true
    });
}


function stopNavigation(){
    if(watchId){
        navigator.geolocation.clearWatch(watchId);
        watchId = null;
    }
}


// ======================
// 🔥 OFF ROUTE
// ======================
function checkOffRoute(user){

    if(!selectedRoute) return;

    let minDist = Infinity;

    selectedRoute.forEach(p => {
        const d = getDistance(user, p);
        if(d < minDist) minDist = d;
    });

    if(minDist > 0.002){
        console.log("⚠️ Off Route");
        reroute(user);
    }
}


// ======================
// 🔄 REROUTE
// ======================
async function reroute(current){

    if(!goalPoint) return;

    const fd = new FormData();

    fd.append("file", base64ToBlob(localStorage.getItem("disasterImage")));

    fd.append("start_x", Math.round(current[1]));
    fd.append("start_y", Math.round(current[0]));

    fd.append("goal_x", Math.round(goalPoint.x));
    fd.append("goal_y", Math.round(goalPoint.y));

    const res = await fetch("http://127.0.0.1:8000/route", {
        method:"POST",
        body:fd
    });

    const data = await res.json();

    if(data.routes){
        drawAllRoutes(data.routes);
        updateLegend(data.routes);
    }

    if(data.geo_enabled && data.geo_routes){
        drawMapRoutes(data.geo_routes);
    }
}


// ======================
// 📏 DISTANCE
// ======================
function getDistance(a, b){
    return Math.sqrt(
        Math.pow(a[0]-b[0],2) +
        Math.pow(a[1]-b[1],2)
    );
}


// ======================
// MAIN INIT
// ======================
function init(){

    const canvas = document.getElementById("resultCanvas");
    const img = new Image();

    img.src = localStorage.getItem("disasterImage");
    canvas._img = img;

    function resizeCanvas(){
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight || 500;
    }

    img.onload = () => {
        resizeCanvas();
        drawImage();
    };

    canvas.addEventListener("click", e => {

        if(routeGenerated) return;

        const rect = canvas.getBoundingClientRect();

        const x = (e.clientX - rect.left - canvas._offsetX) / canvas._scale;
        const y = (e.clientY - rect.top - canvas._offsetY) / canvas._scale;

        if(!startPoint) startPoint = {x,y};
        else if(!goalPoint) goalPoint = {x,y};
        else { startPoint = {x,y}; goalPoint = null; }

        drawImage();
        drawPoint(startPoint, "#00ff88");
        if(goalPoint) drawPoint(goalPoint, "#ff3b3b");
    });

    window.computeRoute = async function(){

        if(!startPoint || !goalPoint){
            alert("Select start & goal");
            return;
        }

        const fd = new FormData();

        fd.append("file", base64ToBlob(localStorage.getItem("disasterImage")));
        fd.append("start_x", Math.round(startPoint.x));
        fd.append("start_y", Math.round(startPoint.y));
        fd.append("goal_x", Math.round(goalPoint.x));
        fd.append("goal_y", Math.round(goalPoint.y));
        fd.append("mode", document.getElementById("routeMode").value);

        // ✅ GEO FIX
        const topLat = localStorage.getItem("top_lat");
        const leftLon = localStorage.getItem("left_lon");
        const bottomLat = localStorage.getItem("bottom_lat");
        const rightLon = localStorage.getItem("right_lon");

        if(topLat && leftLon && bottomLat && rightLon){
            fd.append("top_lat", parseFloat(topLat));
            fd.append("left_lon", parseFloat(leftLon));
            fd.append("bottom_lat", parseFloat(bottomLat));
            fd.append("right_lon", parseFloat(rightLon));
        }

        const res = await fetch("http://127.0.0.1:8000/route", {
            method:"POST",
            body:fd
        });

        const data = await res.json();

        if(data.error){
            alert(data.error);
            return;
        }

        document.getElementById("distance").innerText =
            (data.distance_km ?? "--") + " km";

        document.getElementById("eta").innerText =
            (data.eta_min ?? "--") + " min";

        if(data.routes){
            drawAllRoutes(data.routes);
            updateLegend(data.routes);
            routeGenerated = true;
        }

        if(data.geo_enabled && data.geo_routes){
            drawMapRoutes(data.geo_routes);

            document.getElementById("map").scrollIntoView({
                behavior: "smooth"
            });
        }
    };
}

window.onload = init;


// ======================
function goBack(){
    window.location.href = "index.html";
}


// ======================
// 🔄 RESET
// ======================
function resetSelection(){
    location.reload();
}


// ======================
// 🔧 HELPERS
// ======================
function base64ToBlob(base64){
    const arr = base64.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    const u8 = new Uint8Array(bstr.length);
    for(let i=0;i<bstr.length;i++) u8[i]=bstr.charCodeAt(i);
    return new Blob([u8],{type:mime});
}