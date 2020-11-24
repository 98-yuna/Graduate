function y = graph_figure(signalDir,modulationTypes)
for modType=1:length(modulationTypes)
  subplot(1, 5, modType);

  files = dir(fullfile(signalDir,"*" + string(modulationTypes(modType)) + "*"));

  idx = randi([1 length(files)]);

  load(fullfile(files(idx).folder, files(idx).name), 'N');
  
  plot(real(N), '-'); 
  grid on; 
  axis equal; 
  axis square
  hold on
  plot(imag(N), '-'); 
  grid on; 
  axis equal; 
  axis square
  hold off
  title(string(modulationTypes(modType)));
  xlabel('Signallength'); ylabel('Amplitude')
  xlim([0,length(N)]);  
end
end
