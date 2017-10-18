int coords_to_index(int4 coords, int4 size) {
    return (((coords.z * size.y) + coords.y) * size.x) + coords.x;
}

int coords_to_index_vec4(int4 coords, int4 size) {
    return (((coords.z * size.y) + coords.y) * (size.x / 4)) + coords.x;
}

int is_coords_less(int4 lhs, int4 rhs) {
    return lhs.x < rhs.x & lhs.y < rhs.y & lhs.z < rhs.z;
}

int4 to_img_coords(int4 thread_coords) {
    return (int4) (thread_coords.x * 4, thread_coords.y, thread_coords.z, thread_coords.w);
}

int4 offset_x(int4 img_coords, int x_offset) {
    return (int4) (img_coords.x + x_offset, img_coords.y, img_coords.z, img_coords.w);
}

int2 offset_x_2d(int4 img_coords, int x_offset) {
    return (int2) (img_coords.x + x_offset, img_coords.y);
}

__kernel void raytracing(__global uchar4 * image_buffer,
                       __private int4 img_size,
                       __local uchar4 * local_buffer,
                       __private int voxel_padding_z) {

    int4 thread_coords_global = (int4) (get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 global_size = (int4) (get_global_size(0), get_global_size(1), get_global_size(2), 0);

    int4 thread_coords_local = (int4) (get_local_id(0), get_local_id(1), get_local_id(2), 0);
    int4 local_size = (int4) (get_local_size(0), get_local_size(1), get_local_size(2), 0);

    int4 padded_coords_global = thread_coords_global * (int4) (1, 1, voxel_padding_z, 1);

    int4 img_coords = to_img_coords(padded_coords_global);

    int local_index = coords_to_index(thread_coords_local, local_size);
    int global_index = coords_to_index_vec4(padded_coords_global, img_size);

    if (is_coords_less(img_coords, img_size)) {
        local_buffer[local_index] = image_buffer[global_index];
    } else {
        local_buffer[local_index] = (uchar4) (0, 0, 0, 0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = local_size.z / 2; i > 0; i >>= 1) {
        if (thread_coords_local.z < i) {
            int local_offset = local_size.y * local_size.x * i;
            local_buffer[local_index] = max(local_buffer[local_index], local_buffer[local_index + local_offset]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (thread_coords_local.z == 0) {
        image_buffer[global_index] = local_buffer[local_index];
    }

}
__constant sampler_t input_sampler = CLK_NORMALIZED_COORDS_FALSE |
                                     CLK_ADDRESS_CLAMP |
                                     CLK_FILTER_NEAREST;

int4 transform_coords(int4 coords, float4 mat_0, float4 mat_1, float4 mat_2, float4 mat_3) {
    return (int4) (
            mat_0.s0 * coords.x + mat_0.s1 * coords.y + mat_0.s2 * coords.z + mat_0.s3,
            mat_1.s0 * coords.x + mat_1.s1 * coords.y + mat_1.s2 * coords.z + mat_1.s3,
            mat_2.s0 * coords.x + mat_2.s1 * coords.y + mat_2.s2 * coords.z + mat_2.s3,
            mat_3.s0 * coords.x + mat_3.s1 * coords.y + mat_3.s2 * coords.z + mat_3.s3
        );

}

__kernel void transformation(__read_only image3d_t input_img,
                           __global uchar4 * transformation_buf,
                           __private int4 img_size,
                           __private float4 mat_0,
                           __private float4 mat_1,
                           __private float4 mat_2,
                           __private float4 mat_3)
{
    int4 thread_coords_global = (int4) (get_global_id(0), get_global_id(1), get_global_id(2), 1);
    int4 global_size = (int4) (get_global_size(0), get_global_size(1), get_global_size(2), 0);

    int4 img_coords = to_img_coords(thread_coords_global);
    if (is_coords_less(img_coords, img_size)) {
        int4 coords_0 = transform_coords(offset_x(img_coords, 0), mat_0, mat_1, mat_2, mat_3);
        int4 coords_1 = transform_coords(offset_x(img_coords, 1), mat_0, mat_1, mat_2, mat_3);
        int4 coords_2 = transform_coords(offset_x(img_coords, 2), mat_0, mat_1, mat_2, mat_3);
        int4 coords_3 = transform_coords(offset_x(img_coords, 3), mat_0, mat_1, mat_2, mat_3);

        uint4 color_0 = read_imageui(input_img, input_sampler, coords_0);
        uint4 color_1 = read_imageui(input_img, input_sampler, coords_1);
        uint4 color_2 = read_imageui(input_img, input_sampler, coords_2);
        uint4 color_3 = read_imageui(input_img, input_sampler, coords_3);

        int global_index = coords_to_index_vec4(thread_coords_global, img_size);
        transformation_buf[global_index] = (uchar4) (color_0.x, color_1.x, color_2.x, color_3.x);
    }
}
__kernel void slicing(__global uchar4 * input_img,
                       __write_only image2d_t output_img,
                       __private int4 img_size) {
    int4 thread_coords_global = (int4) (get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 global_size = (int4) (get_global_size(0), get_global_size(1), get_global_size(2), 0);

    int4 img_coords = to_img_coords(thread_coords_global);
    if (is_coords_less(img_coords, img_size)) {
        int global_index = coords_to_index_vec4(thread_coords_global, img_size);
        uchar4 colors_vec = input_img[global_index];

        float4 color_0 = (float4) (colors_vec.s0, colors_vec.s0, colors_vec.s0, 255.0f) / 255.0f;
        float4 color_1 = (float4) (colors_vec.s1, colors_vec.s1, colors_vec.s1, 255.0f) / 255.0f;
        float4 color_2 = (float4) (colors_vec.s2, colors_vec.s2, colors_vec.s2, 255.0f) / 255.0f;
        float4 color_3 = (float4) (colors_vec.s3, colors_vec.s3, colors_vec.s3, 255.0f) / 255.0f;

        write_imagef(output_img, offset_x_2d(img_coords, 0), color_0);
        write_imagef(output_img, offset_x_2d(img_coords, 1), color_1);
        write_imagef(output_img, offset_x_2d(img_coords, 2), color_2);
        write_imagef(output_img, offset_x_2d(img_coords, 3), color_3);
    }
}
