forward_kernel = '''
extern "C" 
__global__ void roi_forward(
	const float* const feat, 
	const float* const roi, 
	float* out, 
	int* max_id,
	const double scale,
	const int c, const int h, const int w,
	const int out_h, const int out_w,
	const int num_out)
{
	#define  MAX(x,y)    ((x) > (y) ? (x) : (y))
	#define  MIN(x,y)    ((x) < (y) ? (x) : (y))
	// process ONE pooling section, i.e., only ouput one pooled number (maximum of this region) 
	// 'roi' consists of coordinates on original image (not feature map)
	// 'feat' consists of coordinates on feature map BEFORE pooling
	// 'pooled' consists of coordinates on feature map AFTER pooling

	int idx = blockIdx.x * blockDim.x + threadIdx.x;  // idx_th pooling section
	if (idx > num_out) return;

	// pooling positions of shape(N, C, H, W), hence handle W first, then H, C, N
	const int pooled_w_th = idx % out_w;  // w_th position of a row on pooled feat map
	const int pooled_h_th = (idx / out_w) % out_h;  // h_th position of a col on pooled feat map
	const int c_th = (idx / out_w / out_h) % c;  // c_th channel
	// num_th roi of all batches, =idx / (all outputs of one single roi)
	int num = idx / (out_w * out_h * c);

	// the start of num_th roi
	const int ptr = num * 5;  
	const int roi_batch_id = roi[ptr + 0];
	// x_min, y_min, x_max, y_max * scale --> x, y, x, y on feature map
	const int feat_w_start = round(roi[ptr + 1] * scale);
	const int feat_h_start = round(roi[ptr + 2] * scale);
	const int feat_w_end = round(roi[ptr + 3] * scale);
	const int feat_h_end = round(roi[ptr + 4] * scale);

	// in case w or h is 0
	const int feat_w = MAX(feat_w_end - feat_w_start, 1);
	const int feat_h = MAX(feat_h_end - feat_h_start, 1);
	// pooling stride of w and h, note that we don't floor or ceil here
	const float stride_w = static_cast<float>(feat_w) / static_cast<float>(out_w);
	const float stride_h = static_cast<float>(feat_h) / static_cast<float>(out_h);

	// (w, h) position on the feature map BEFORE pooling
	// the section of the feature map to be pooled here
	// (pooled_w_th, pooled_h_th) is position on the feature map AFTER pooling
	int w_th_feat_start = static_cast<int>(floor(static_cast<float>(pooled_w_th) * stride_w));
	int h_th_feat_start = static_cast<int>(floor(static_cast<float>(pooled_h_th) * stride_h));
	int w_th_feat_end = static_cast<int>(ceil(static_cast<float>(pooled_w_th + 1) * stride_w));
	int h_th_feat_end = static_cast<int>(ceil(static_cast<float>(pooled_h_th + 1) * stride_h));

	// clip down to within the feature map (before pooling)
	w_th_feat_start = MIN(MAX(w_th_feat_start, 0), w);
	h_th_feat_start = MIN(MAX(h_th_feat_start, 0), h);
	w_th_feat_end = MIN(MAX(w_th_feat_end, 0), w);
	h_th_feat_end = MIN(MAX(h_th_feat_end, 0), h);

	// if nothing is pooled, set output to 0, and idx to -1 so that no grad back prop
	bool empty = (w_th_feat_end <= w_th_feat_start) || (h_th_feat_end <= h_th_feat_start);
	float max_feat = empty ? 0 : -1E+37;
	int max_idx = -1;

	// pooling
	// the position of the roi to which the section currently processing belongs starts from
	const int roi_data_offset = (roi_batch_id * c + c_th) * h * w;
	for (int h_i = h_th_feat_start; h_i < h_th_feat_end; ++h_i)
	{
		for (int w_j = w_th_feat_start; w_j < w_th_feat_end; ++w_j)
		{
			int feat_idx = h_i * w + w_j;
			if (feat[roi_data_offset + feat_idx] > max_feat)
			{
				max_feat = feat[roi_data_offset + feat_idx];
				max_idx = feat_idx;
			}
		}
	}
	out[idx] = max_feat;
	max_id[idx] = max_idx;
}
'''
