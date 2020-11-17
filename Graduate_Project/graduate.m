% --------[ 5개의 변조 유형(연속신호 디지털 변조)을 categorial형 배열(1X5)로 생성 ]--------
modulationTypes = categorical(["BPSK", "QPSK", "8PSK","16QAM", "64QAM"]); 
 
% ----------------------[ 학습데이터 생성하기 ]----------------------
% 학습 신호데이터 2000개 생성
% 1개의 신호(프레임)는 1024 sample로 이루어져 있고, 각 샘플은 8bits로 이루어져있음
% 1 symbol = 8 sample => 하나의 신호는 256 symbols
numFramesPerModType = 2000;

sampleperSymbol = 8; 
sampleperFrame = 1024;  

% 샘플링레이트 = 200kHz
% 중심주파수 = 2.4GHz

% cf) 중심주파수(Intermediate Frequency)
% 실제로 통신 시스템에서 정보를 담은 주파수는 낮은 기저대역(baseband) 주파수이다. 
% 이러한 베이스밴드 주파수대의 신호는, 실제로 안테나를 떠나 전자파로서 방출하기 위한 carrier 주파수로 올려서 보내야한다. 
% 신호전송을 위한 carrier 주파수는 각각의 통신시스템이 사용하는 주파수를 말한다.
% baseband 주파수를 carrier 주파수로 바로 변환하는 것이 아니라, 
% 중간에 다른 주파수로 한번 변환한 후, 다시 이 주파수를 carrier로 변환하는데 그 중간의 주파수를 중간주파수라고 한다. 
% 거의 대부분의 통신시스템은 IF 단을 가지는 구조이며, 이것을 헤테로다인(heterodyne)방식이라고 한다. 

% cf) 무선주파수 대역폭 : 3kHz ~ 300GHz
% 무선랜은 가장 널리 사용되는 무선 네트워크 기술로 현재 2.4GHz 대역에서 상용화되어 있기 때문에 2.4GHz를 중심 주파수로
% 정했음
fs = 200e3; 
fc = 2.4e9; 

% -------------[ 생성한 입력신호가 지나갈 채널을 만들기]------------
% 실제 수신환경과 비슷하게 만들기 위해 다중경로 페이딩채널을 고려함
% cf) 다중경로 페이딩(Multipath Fading)
% 서로 다른 경로를 따라 수신된 전파들이 여러 물체에 의한 다중 반사 등으로 인해
% 서로 다른 진폭, 위상, 입사각, 편파 등이 간섭을 일으켜 불규칙하게 요동치는 현상
% 다중경로 페이딩을 겪으면서 채널은 각 확산과 시간지역 확산 그리고 도플러 확산을 일으킴

reyleighchan = comm.RayleighChannel('SampleRate', fs, 'PathDelays',[0 1.5e-4], ...
    'AveragePathGains',[2 3],'MaximumDopplerShift',4 );

% f = fullfile(폴더명, 파일이름) = 지정된 폴더 및 파일 이름만들기 
% ex) f = fullfile('myfolder', 'mysubfolder', 'myfile.m') 인 경우
%     f = 'myfolder/mysubfolder/myfile.m
% 나는 tmp 디렉터리 안에 ModulationDataFiles를 생성하겠다.
predictDir = fullfile(tempdir,"ModulationFiles");
% "C:\Users\USER\AppData\Local\Temp\ModulationFiles" 들어가면 신호파일 확인가능

% 저장된 신호들의 이름시작은 signal로 
fileNameRoot = "signal";
numModulationTypes = length(modulationTypes); % = 5

% 입력신호파일이 존재하는가?
signalFilesExist = false; 

% predictDir이 존재하는 경우 -> 디렉터리 읽어오기 
if exist(predictDir,'dir')
  files = ls(fullfile(predictDir,sprintf("%s*",fileNameRoot)));
  if length(files) == numModulationTypes*numFramesPerModType
      signalFilesExist = true;
  end
end

tic

% 파일이 존재하지 않는경우
if ~signalFilesExist
    [success, msg, msgID] =  mkdir(predictDir);
    % 만약, 폴더 생성에 실패한 경우, 왜 오류가 발생했는지 메세지를 출력한다.
    if ~success
        error(msgID,msg)
    end
  % modType = 1:5까지 반복문을 돌려 입력신호에 넣을 파형을 생성한다. 
  for modType = 1: numModulationTypes
      fprintf('%s 입력파일생성중 %s frames\n',datestr(toc/86400,'HH:MM:SS'), modulationTypes(modType))
      % 프레임이 생성되고있는중 배열1~5번까지
      label = modulationTypes(modType);  
      % create_rand 함수를 호출하여 정수형 난수발생
      data_rand = create_rand(modulationTypes(modType));
      % 변조신호 하나당 2000개의 샘플 생성
      for p=1:numFramesPerModType
      % 생성된 랜덤난수로 변조신호 생성하기
      x = data_rand();
      % 256 x 1의 열벡터 x를 modulate함수를 호출하여 변조
      layer = modulate(modulationTypes(modType), data_rand);
      % 생성된 변조신호를 채널을 통과시켜 다중모드를 고려한 신호를 생성
      channel_pass = reyleighchan(layer);
      % 데이터 정규화 
      N = normalize(channel_pass,'zscore', 'std');
      
      % 파일 저장하기 파일이름은 signal+변조유형+신호갯수(?)숫자
      fileName = fullfile(predictDir,sprintf("%s%s%d",fileNameRoot,modulationTypes(modType),p));
      save(fileName,"N","label");
     end  
  end
else
  disp("메모리에 저장되어 있는 데이터파일이 존재합니다.")
  disp("메모리에 저장되어있는 신호파일을 불러옵니다.")
end

% 내가 생성한 신호의 그래프를 보고싶다면?
graph_figure(predictDir,modulationTypes);
% 저장된 신호를 수집해오는 signalDatastore 객체 생성
datastore1 = signalDatastore("C:\Users\user\AppData\Local\Temp\ModulationFiles",'SignalVariableNames',["label","N"])

% 데이터 스토어에 저장되어있는 신호 변환 transform(signalDatastore객체, 익명함수)
input_signal = transform(datastore1,@ input_trans);

% 생성한 신호데이터를 80%는 학습용, 20%는 테스트용으로 분리 
percentTrainingSamples = 80;
percentTestSamples = 20;
rxLabels=gather(tall(transform(input_signal, @import_label)));
modulationTypes = unique(string(rxLabels));

numFrames = length(rxLabels);
numTrainFrames = round(numFrames*percentTrainingSamples/100);
numTestFrames = round(numFrames*percentTestSamples/100);

trainIdx = zeros(numTrainFrames,1);
testIdx = zeros(numTestFrames,1);
trainFrameCnt = 0;
testFrameCnt = 0;

for modType = 1:length(modulationTypes)
  rawIdx = find(string(rxLabels) == modulationTypes(modType));
  numFrames = length(rawIdx);
  
  numTrainFrames = round(numFrames*percentTrainingSamples/100);
  numTestFrames = round(numFrames*percentTestSamples/100);
  
  trainIdx((1:numTrainFrames)+trainFrameCnt,1) = rawIdx(1:numTrainFrames,1);
  trainFrameCnt = trainFrameCnt + numTrainFrames;
  
  testIdx((1:numTestFrames)+testFrameCnt,1) = rawIdx(1:numTestFrames,1);
  testFrameCnt = testFrameCnt + numTestFrames;
end

trainDS = subset(input_signal, trainIdx);
testDS = subset(input_signal, testIdx);
input_signal=trainDS;
test_signal=testDS;

% 2차원 데이터를 분류하기 위해서는 입력데이터가 4차원 형태여야한다. 

test_label= gather(tall(transform(test_signal, @import_label)));
test_frame = gather(tall(transform(test_signal, @import_frame)));
% 2차원 데이터를 분류하기 위해서는 입력데이터가 4차원 형태여야한다. 
testsignal = cat(4, test_frame{:}); 
% 응답변수는 categorical형 변수여야하므로 형변환시키기 -> 그렇지않으면 오류가 발생함
testlabel = categorical(string(test_label));


input_label = gather(tall(transform(input_signal, @import_label)));
input_frame = gather(tall(transform(input_signal, @import_frame)));
inputsignal = cat(4, input_frame{:}); 
inputlabel = categorical(string(input_label));
% 입력데이터의 형태를 보면 1x1024(1024sample로 구성)x2(실수부,허수부)x10000(신호갯수)
size(inputsignal)

filterSize = [1 4];
poolSize = [1 4];
dropoutRate = 0.5;

layers = [ 
    % 데이터 정규화를 이미 한 값이므로 데이터정규화 x
    imageInputLayer([1 1024 2], 'Normalization', 'none', 'Name', 'Input Layer')
    convolution2dLayer([1 4], 5, 'Padding', 'same', 'Name', 'CNN1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'ReLU1')
    maxPooling2dLayer(poolSize, 'Stride', [1 4], 'Name', 'MaxPool1')
    
    convolution2dLayer(filterSize, 10, 'Padding', 'same', 'Name', 'CNN2')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'ReLU2')
    maxPooling2dLayer(poolSize, 'Stride', [1 4], 'Name', 'MaxPool2')
  
    
    convolution2dLayer(filterSize, 20, 'Padding', 'same', 'Name', 'CNN3')
    batchNormalizationLayer('Name', 'BN3')
    reluLayer('Name', 'ReLU3')
    maxPooling2dLayer(poolSize, 'Stride', [1 4], 'Name', 'MaxPool3')
    
      
    convolution2dLayer(filterSize, 40, 'Padding', 'same', 'Name', 'CNN4')
    batchNormalizationLayer('Name', 'BN4')
    reluLayer('Name', 'ReLU4')
    maxPooling2dLayer(poolSize, 'Stride', [1 4], 'Name', 'MaxPool4')

    fullyConnectedLayer(numModulationTypes, 'Name', 'FC1')
    softmaxLayer('Name', 'SoftMax')
    classificationLayer('Name', 'Output') ];

% layers로 지정한 딥러닝 신경망 분석
% 신경망 계층에 대한 정보 제공
analyzeNetwork(layers);

options = trainingOptions('sgdm','MaxEpochs',15, 'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.0001 ,'Plots','training-progress')

% 신경망을 훈련시키려면 훈련옵션을 trainNetwork 함수의 입력 인수로 사용
% 확률적경사하강법을 이용한 신경망 훈련
net = trainNetwork(inputsignal,inputlabel,layers, options)

% 학습된 신경망을 가지고 데이터예측해보기 
Pred=classify(net,inputsignal); 
Accuracy=mean(Pred == inputlabel);
disp("Train accuracy: " + Accuracy*100 + "%")

% 테스트셋 정확도 = 95.75%
Pred2=predict(net, testsignal, 'ReturnCategorical',true);
Accuracy2=mean(Pred2 == testlabel);
disp("Test accuracy: " + Accuracy2*100 + "%")

% 어느부분에서 데이터가 혼동이 되는지 표로 출력하기
figure
cm= confusionchart(testlabel, Pred2);
cm.Title= '테스트 데이터의 혼동 행렬 표';
cm.RowSummary = 'row-normalized';
cm.Parent.Position = [cm.Parent.Position(1:2) 740 424];

