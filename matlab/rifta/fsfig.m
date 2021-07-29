function HF = fsfig(Name)

if nargin<1
    Name = '';
end



HF = figure('Name', Name, 'Color', 'w');
sz=get(0,'MonitorPositions');
set(HF,'OuterPosition',sz(1,:),'Resize','on');
warning('off','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');
J = get(HF,'javaframe');
drawnow;
set(J,'maximized',true);
warning('on','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');

end
