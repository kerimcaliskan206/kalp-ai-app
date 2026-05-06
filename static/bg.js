(function () {
    var canvas = document.getElementById('networkCanvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var W = 0, H = 0, hearts = [], t = 0;
    function hx(a) { return 16 * Math.pow(Math.sin(a), 3); }
    function hy(a) { return -(13*Math.cos(a)-5*Math.cos(2*a)-2*Math.cos(3*a)-Math.cos(4*a)); }
    var TPL = Array.from({length:28}, function(_,i){ var a=(i/28)*Math.PI*2; return {lx:hx(a),ly:hy(a)}; });
    function makeHeart(fx,fy){ var d=0.2+Math.random()*0.8,s=0.10+Math.random()*0.20; return {x:fx!==undefined?fx:Math.random()*W,y:fy!==undefined?fy:Math.random()*H,vx:(Math.random()-0.5)*s,vy:-(0.07+Math.random()*0.16),rot:Math.random()*Math.PI*2,rotV:(Math.random()-0.5)*0.007,size:9+Math.random()*17,depth:d,alpha:0.10+d*0.22,breathe:Math.random()*Math.PI*2,breatheS:0.014+Math.random()*0.01,wobble:Math.random()*Math.PI*2,wobbleS:0.018+Math.random()*0.014}; }
    function setSize(){ W=canvas.width=window.innerWidth; H=canvas.height=window.innerHeight; }
    function init(){ setSize(); hearts=Array.from({length:7},function(){ return makeHeart(Math.random()*W,Math.random()*H); }); }
    function onResize(){ setSize(); hearts.forEach(function(h){ h.x=Math.random()*W; h.y=Math.random()*H; }); }
    function drawHeart(h){ var br=1+Math.sin(t*h.breatheS+h.breathe)*0.06,scale=(h.size/17)*br,cosR=Math.cos(h.rot),sinR=Math.sin(h.rot); for(var i=0;i<TPL.length;i++){ var j=(i+1)%TPL.length,pi=TPL[i],pj=TPL[j]; ctx.beginPath(); ctx.moveTo(h.x+(pi.lx*cosR-pi.ly*sinR)*scale,h.y+(pi.lx*sinR+pi.ly*cosR)*scale); ctx.lineTo(h.x+(pj.lx*cosR-pj.ly*sinR)*scale,h.y+(pj.lx*sinR+pj.ly*cosR)*scale); ctx.strokeStyle='rgba(200,18,45,'+(h.alpha*0.32)+')'; ctx.lineWidth=0.4; ctx.stroke(); } TPL.forEach(function(p,i){ var rx=p.lx*cosR-p.ly*sinR,ry=p.lx*sinR+p.ly*cosR,sx=h.x+rx*scale,sy=h.y+ry*scale,pulse=0.5+0.5*Math.sin(t*0.04+i*0.4+h.breathe),r=(0.8+h.depth*1.4)*(0.85+0.2*pulse),a=h.alpha*(0.6+0.4*pulse); if(h.depth>0.6&&pulse>0.7){ ctx.beginPath(); ctx.arc(sx,sy,r*2.5,0,Math.PI*2); ctx.fillStyle='rgba(220,20,50,'+(a*0.16)+')'; ctx.fill(); } ctx.beginPath(); ctx.arc(sx,sy,r,0,Math.PI*2); ctx.fillStyle='rgba(225,28,55,'+a+')'; ctx.fill(); }); }
    function update(){ hearts.forEach(function(h){ h.x+=h.vx+Math.sin(t*h.wobbleS+h.wobble)*0.28; h.y+=h.vy; h.rot+=h.rotV; if(h.y<-h.size*3){h.y=H+h.size*3;h.x=Math.random()*W;} if(h.x<-h.size*4)h.x=W+h.size*4; if(h.x>W+h.size*4)h.x=-h.size*4; }); }
    function draw(){ ctx.clearRect(0,0,W,H); t++; update(); hearts.forEach(drawHeart); requestAnimationFrame(draw); }
    window.addEventListener('resize',onResize);
    function start(){ init(); draw(); }
    if(document.readyState==='complete') start(); else window.addEventListener('load',start);
})();