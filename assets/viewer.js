
(function(){
  const modal=document.getElementById('viewer');
  if(!modal) return;
  const img=document.getElementById('viewer-image'),title=document.getElementById('viewer-title'),
        openLink=document.getElementById('viewer-open');
  function close(){modal.classList.remove('open');img.src='';document.body.style.overflow='';}
  document.querySelectorAll('[data-src]').forEach(function(el){
    el.addEventListener('click',function(){
      img.src=el.dataset.src;img.alt=el.dataset.title;title.textContent=el.dataset.title;
      openLink.href=el.dataset.src;modal.classList.add('open');document.body.style.overflow='hidden';
    });
  });
  const closeBtn=document.getElementById('viewer-close');
  if(closeBtn) closeBtn.addEventListener('click',close);
  modal.addEventListener('click',function(e){if(e.target===modal)close();});
  document.addEventListener('keydown',function(e){if(e.key==='Escape')close();});
  document.querySelectorAll('.filter').forEach(function(btn){
    btn.addEventListener('click',function(){
      document.querySelectorAll('.filter').forEach(function(b){b.classList.remove('active');});
      btn.classList.add('active');
      const f=btn.dataset.filter;
      document.querySelectorAll('.stage').forEach(function(s){
        s.style.display=(f==='all'||s.dataset.category===f)?'grid':'none';
      });
    });
  });
})();
