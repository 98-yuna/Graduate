function ModData = modulate(modType,data_rand)
 switch modType
  case 'BPSK'
      ModData = bpskModulator(data_rand);
  case 'QPSK'
      ModData = qpskModulator(data_rand);
  case '8PSK'
      ModData  = psk8Modulator(data_rand);
  case '16QAM'
      ModData = qam16Modulator(data_rand);
  case '64QAM'
      ModData  = qam64Modulator(data_rand);
end
end
function y = bpskModulator(data_rand)
channel = comm.AWGNChannel();
% bpsk이므로 pskmod(입력데이터, 변조지수)
syms = pskmod(data_rand,2); 
channelOutput = channel(syms);
% 올림코사인 FIR 펄스 성형필터
% rcosdesign(롤오프인자,잘리는 심볼갯수, 각 심볼 주기에 포함되는 샘플)
% a = rcosdesign(0.35, 4, 8);
% 펄스 성형을 위해 sample rate 8배 증가시키기(upsampling)
% 업샘플링기법 = 처리할 신호를 원 신호의 샘플링보다 더 높은 sampling rate로 변경하여
% 필터링 회로를 쉽게 구현할 수 있게 한다.
y = filter(rcosdesign(0.35, 4, 8), 1, upsample(channelOutput,4));
end

function y = qpskModulator(data_rand)
channel = comm.AWGNChannel();
% qpsk이므로 pskmod(입력데이터, 변조지수, 45도)
syms = pskmod(data_rand,4,pi/4);
channelOutput = channel(syms);
y = filter(rcosdesign(0.35, 4, 8), 1, upsample(channelOutput,4));
end

function y = psk8Modulator(data_rand)
channel = comm.AWGNChannel();
% 8psk이므로 pskmod(입력데이터, 8)
syms = pskmod(data_rand,8, pi/8);
channelOutput = channel(syms);
y = filter(rcosdesign(0.35, 4, 8), 1, upsample(channelOutput,4));
end

function y = qam16Modulator(data_rand)
channel = comm.AWGNChannel();
syms = qammod(data_rand,16);
channelOutput = channel(syms);
y = filter(rcosdesign(0.35, 4, 8), 1, upsample(channelOutput,4));
end

function y = qam64Modulator(data_rand)
channel = comm.AWGNChannel();
syms = qammod(data_rand,64);
channelOutput = channel(syms);
y = filter(rcosdesign(0.35, 4, 8), 1, upsample(channelOutput,4));
end