function y = input_trans(in)

frame_label = in{1};
frame = in{2};
i = real(frame);
q = imag(frame);
% 전치행렬을 하는이유 : 입력데이터 형식이 1xAxKxN의 형태로 행부분이 1이어야한다.
% i는 2048x1형태의 배열이므로 전치행렬을 취해야한다. 
I = transpose(i);
Q = transpose(q);

% 실수부와 허수부부분을 분리하여 3차원배열로 생성 
frame_real = cat(3,I,Q);
y = {frame_label, frame_real};
end