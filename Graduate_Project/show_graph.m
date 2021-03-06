% -------- BPSK그래프 출력하기 --------
data1 = load("C:\Users\USER\AppData\Local\Temp\ModulationFiles\signalBPSK1000");
graph1 = downsample(data1.layer ,8)
 
figure
plot(real(graph1), '.')
title('BPSK')

% -------- QPSK그래프 출력하기 --------
data2 = load("C:\Users\USER\AppData\Local\Temp\ModulationFiles\signalQPSK1000");
graph2 = downsample(data2.layer,8)

figure
plot(real(graph2), imag(graph2),'o')
title('QPSK')

% -------- 8PSK그래프 출력하기 --------
data3 = load("C:\Users\USER\AppData\Local\Temp\ModulationFiles\signal8PSK1000");
graph3 = downsample(data3.layer ,8)
 
figure
plot(real(graph3), imag(graph3),'o')
title('8PSK')

% -------- 16QAM그래프 출력하기 --------
data4 = load("C:\Users\USER\AppData\Local\Temp\ModulationFiles\signal16QAM1000");
graph4 = downsample(data4.layer ,8)

figure
plot(real(graph4), imag(graph4), '.')
title('16QAM')

% -------- 64AM그래프 출력하기 --------
data5 = load("C:\Users\USER\AppData\Local\Temp\ModulationFiles\signal64QAM1000");
graph5 = downsample(data5.layer ,8)

figure
plot(real(graph5), imag(graph5), 'o')
title('64QAM')