function result = create_rand(modType)
 switch modType
  case "BPSK"
    % [0 1]구간에서 정수형 난수로 구성된 256 x 1 열벡터생성
    % 256개의 symbol을 생성한다 => 1 symbol은 8 sample로 구성 => 1024개의 sample 생성 
    % 1 frame 생성
   result = randi([0 1],256,1);
  case "QPSK"
    result = randi([0 3],256,1);
  case "8PSK"
    result = randi([0 7],256,1);
  case "16QAM"
    result = randi([0 15],256,1);
  case "64QAM"
   result = randi([0 63],256,1);
end
end