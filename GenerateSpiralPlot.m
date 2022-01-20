p = load('SpiralPlotPoints.txt');
cl = p(:,3);
plot(p(cl==1,1), p(cl==1,2),'bx');
hold on
plot(p(cl==-1,1), p(cl==-1,2),'ro');
cdat = load('SpiralPlotContour.txt');
contour(linspace(0,1,251),linspace(0,1,251), cdat',[0 0],'k--');
xlabel('x_1')
ylabel('x_2')