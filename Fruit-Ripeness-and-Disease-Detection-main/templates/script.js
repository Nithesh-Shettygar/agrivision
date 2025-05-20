// Toggle side menu
function toggleMenu() {
    const sideMenu = document.getElementById('sideMenu');
    const mainContent = document.getElementById('mainContent');
    
    sideMenu.classList.toggle('active');
    mainContent.classList.toggle('shifted');
}

// Close menu when clicking outside
document.addEventListener('DOMContentLoaded', function() {
    document.addEventListener('click', function(event) {
        const sideMenu = document.getElementById('sideMenu');
        const menuToggle = document.querySelector('.menu-toggle');
        
        if (!sideMenu.contains(event.target) && event.target !== menuToggle) {
            sideMenu.classList.remove('active');
            document.getElementById('mainContent').classList.remove('shifted');
        }
    });
});