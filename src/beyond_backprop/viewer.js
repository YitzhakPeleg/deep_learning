() => {
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft')
            document.querySelector('#prev-btn button')?.click();
        if (e.key === 'ArrowRight')
            document.querySelector('#next-btn button')?.click();
    });
}
