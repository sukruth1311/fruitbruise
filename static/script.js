document.addEventListener('DOMContentLoaded', function() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.querySelector('input[type="file"]');

    // Highlight dropzone when file is dragged over
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('highlight');
    }

    function unhighlight(e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('highlight');
    }

    // Handle dropped files
    dropzone.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;

        // Preview image
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    dropzone.style.backgroundImage = `url(${e.target.result})`;
                    dropzone.querySelector('p').textContent = file.name;
                    dropzone.querySelector('span').textContent = `${(file.size/1024/1024).toFixed(2)} MB`;
                }
                reader.readAsDataURL(file);
            }
        }
    });

    // Handle file selection via click
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const file = this.files[0];
            const reader = new FileReader();
            reader.onload = function(e) {
                dropzone.style.backgroundImage = `url(${e.target.result})`;
                dropzone.querySelector('p').textContent = file.name;
                dropzone.querySelector('span').textContent = `${(file.size/1024/1024).toFixed(2)} MB`;
            }
            reader.readAsDataURL(file);
        }
    });
});