// ======================
// 🚀 GO TO RESULT PAGE
// ======================
window.goToResult = async function(){

    const roadFile = document.getElementById("roadInput").files[0];
    const disasterFile = document.getElementById("disasterInput").files[0];

    // ======================
    // ✅ VALIDATION
    // ======================
    if(!roadFile || !disasterFile){
        alert("Upload BOTH images");
        return;
    }

    // ======================
    // 🔄 BASE64 CONVERTER
    // ======================
    function toBase64(file){
        return new Promise((resolve, reject)=>{
            const reader = new FileReader();

            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);

            reader.readAsDataURL(file);
        });
    }

    try{
        // ======================
        // 📦 CONVERT IMAGES
        // ======================
        const roadBase64 = await toBase64(roadFile);
        const disasterBase64 = await toBase64(disasterFile);

        // ======================
        // 💾 STORE IMAGES
        // ======================
        localStorage.setItem("roadImage", roadBase64);
        localStorage.setItem("disasterImage", disasterBase64);

        // ======================
        // 🌍 GEO INPUT (OPTIONAL)
        // ======================
        const topLatRaw = document.getElementById("top_lat")?.value;
        const leftLonRaw = document.getElementById("left_lon")?.value;
        const bottomLatRaw = document.getElementById("bottom_lat")?.value;
        const rightLonRaw = document.getElementById("right_lon")?.value;

        console.log("🌍 GEO INPUT RAW:", topLatRaw, leftLonRaw, bottomLatRaw, rightLonRaw);

        // ======================
        // 🔍 SAFE PARSE
        // ======================
        const topLat = parseFloat(topLatRaw);
        const leftLon = parseFloat(leftLonRaw);
        const bottomLat = parseFloat(bottomLatRaw);
        const rightLon = parseFloat(rightLonRaw);

        const geoValid =
            !isNaN(topLat) &&
            !isNaN(leftLon) &&
            !isNaN(bottomLat) &&
            !isNaN(rightLon);

        // ======================
        // ✅ STORE GEO IF VALID
        // ======================
        if(geoValid){

            localStorage.setItem("top_lat", topLat);
            localStorage.setItem("left_lon", leftLon);
            localStorage.setItem("bottom_lat", bottomLat);
            localStorage.setItem("right_lon", rightLon);

            console.log("✅ GEO STORED:", {
                topLat, leftLon, bottomLat, rightLon
            });

        } else {

            // 🔥 REMOVE OLD VALUES
            localStorage.removeItem("top_lat");
            localStorage.removeItem("left_lon");
            localStorage.removeItem("bottom_lat");
            localStorage.removeItem("right_lon");

            console.warn("⚠️ GEO INVALID OR EMPTY → MAP DISABLED");
        }

        // ======================
        // 🧹 CLEAN OLD DATA
        // ======================
        localStorage.removeItem("startPoint");
        localStorage.removeItem("goalPoint");

        // ======================
        // 🚀 REDIRECT
        // ======================
        window.location.href = "result.html";

    } catch(err){
        console.error("❌ ERROR:", err);
        alert("Error processing images");
    }
};