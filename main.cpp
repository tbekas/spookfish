#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cstdarg>
#include <iterator>
#include <string.h>
#include <GL/glew.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <GL/freeglut.h>
#include <CL/cl_platform.h>
#include <sstream>
#include <ctime>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmjpeg/djdecode.h>
#include <boost/program_options.hpp>

#define RAYTRACING_KERNEL "raytracing.cl"
#define TRANSFORMATION_KERNEL "transformation.cl"
#define SLICING_KERNEL "slicing.cl"
#define VERTEX_SHADER "shader.vert"
#define FRAGMENT_SHADER "shader.frag"

#define MAIN_TEXTURE_UNIT_INDEX 1

#define div_up(a, b) ((a) + (b) - 1) / (b)

#define round_up(a, up) div_up((a), (up)) * (up)
#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

using namespace std;

// This program targets first device on first platform

void init_cl();

cl_device_id get_device(const cl_platform_id platform, unsigned int device_num);

cl_context get_context(const cl_platform_id, const cl_device_id device);

cl_platform_id get_platform(unsigned int platform_num);

cl_program get_cl_program(const cl_context context, const string filename, const string filename2, const string filename3);

vector<char> read_file_to_buffer(const string filename);

string read_file(const string filename);

vector<float> to_floats(const vector<char> data);

void build_cl_program(const cl_program program, cl_device_id const device);

void error(const string error_message, ...);

void error_check(cl_int err, const string error_message, ...);

void error_check(int err);

void init_glut(int &argc, char *argv[]);

cl_mem create_input_image_object(const cl_context context, const unsigned char *input_image_array);

cl_mem create_transformation_buffer(const cl_context context, const unsigned char *transformation_buffer_array);

cl_mem create_output_image_object(const cl_context context, const float * output_array);

void init_gl();

void print_info();

void print_time_diff(clock_t start);

streamsize get_stream_size(ifstream &input_file);

GLuint create_gl_program(GLuint vertex_shader, GLuint fragment_shader);

GLuint create_shader(GLenum shader_type, const string shader_file) ;

GLuint create_texture() ;

GLuint create_sampler() ;

void draw_texture();

void prepare_texture();

void print_work_size(const char *message, size_t *work_size);

GLuint position_buffer_obj;
GLuint vao;
GLuint program;
GLuint texture;

GLuint sampler;
cl_command_queue queue;
cl_kernel raytracing_kernel;
cl_kernel transformation_kernel;
cl_kernel slicing_kernel;
size_t default_offset[3] = {0, 0, 0};
size_t *default_local_work_size;
size_t *default_global_work_size;
size_t *slicing_global_work_size;
size_t *raytracing_local_work_size;

cl_mem output_image;
int width;
int height;

int depth;
int mouse_active_click;
int mouse_left_x;
int mouse_left_y;
int transformation_x = WINDOW_WIDTH / 2;

int transformation_y = WINDOW_HEIGHT / 2;
int mouse_right_x;
int mouse_right_y;
int slice_number = 0;
bool slice_view_mode = false;

int mouse_wheel_y = 0;

const float vertex_positions[] = {
        // vertex coords
        -1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 1.0f,

        // texture coords
        0.0f, 1.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f,
};

cl_int err;

void display()
{
    prepare_texture();

    draw_texture();

    glutSwapBuffers();

    glutPostRedisplay();
}

void prepare_texture() {

    glFinish();

    int window_width = glutGet(GLUT_WINDOW_WIDTH);
    int window_height = glutGet(GLUT_WINDOW_HEIGHT);

    float angle_x = ((float) transformation_y / (float) window_height - 0.5f) * (float) M_PI;
    float angle_y = ((float) transformation_x / (float) window_width - 0.5f) * (float) M_PI;

    glm::mat4 trans;

    trans *= glm::translate(glm::mat4(1.0f), glm::vec3(width * 0.5f, height * 0.5f, depth * 0.5f));
    trans *= glm::rotate(glm::mat4(1.0f), angle_x, glm::vec3(1.0f, 0.0f, 0.0f));
    trans *= glm::rotate(glm::mat4(1.0f), angle_y, glm::vec3(0.0f, 1.0f, 0.0f));
    trans *= glm::translate(glm::mat4(1.0f), glm::vec3(-width * 0.5f, -height * 0.5f, -depth * 0.5f));

    const float * transformation_matrix = glm::value_ptr(glm::transpose(trans));
    err = clSetKernelArg(transformation_kernel, 3, sizeof(float) * 4, transformation_matrix);
    error_check(err);
    err = clSetKernelArg(transformation_kernel, 4, sizeof(float) * 4, transformation_matrix+4);
    error_check(err);
    err = clSetKernelArg(transformation_kernel, 5, sizeof(float) * 4, transformation_matrix+8);
    error_check(err);
    err = clSetKernelArg(transformation_kernel, 6, sizeof(float) * 4, transformation_matrix+12);
    error_check(err);

    if (slice_view_mode) {
        size_t offset[3] = {0 ,0, slice_number};
        err = clEnqueueNDRangeKernel(queue, transformation_kernel, 3,
                                     offset, slicing_global_work_size, default_local_work_size,
                                     NULL, 0, NULL);
        error_check(err);
    } else {
        err = clEnqueueNDRangeKernel(queue, transformation_kernel, 3,
                                     default_offset, default_global_work_size, default_local_work_size,
                                     NULL, 0, NULL);
        error_check(err);
    }

    err = clEnqueueAcquireGLObjects(queue, 1, &output_image, 0, NULL, NULL);
    error_check(err);

    if (slice_view_mode) {
        size_t offset[3] = {0 ,0, slice_number};
        err = clEnqueueNDRangeKernel(queue, slicing_kernel, 3,
                                     offset, slicing_global_work_size, default_local_work_size,
                                     NULL, 0, NULL);
        error_check(err);
    } else {
        int voxel_padding = 1;
        int i = 0;
        while(voxel_padding < depth) {
            err = clSetKernelArg(raytracing_kernel, 3, sizeof(int), &voxel_padding);
            error_check(err);

            size_t global_work_size[3] = {round_up(width / 4, raytracing_local_work_size[0]),
                                          height, round_up(div_up(depth, voxel_padding),
                                          raytracing_local_work_size[2])};
            err = clEnqueueNDRangeKernel(queue, raytracing_kernel, 3,
                                         default_offset,
                                         global_work_size,
                                         raytracing_local_work_size,
                                         NULL, 0, NULL);
            error_check(err);

            voxel_padding *= 4;
        }

        err = clEnqueueNDRangeKernel(queue, slicing_kernel, 3,
                                     default_offset, slicing_global_work_size, default_local_work_size,
                                     NULL, 0, NULL);
        error_check(err);
    }

    err = clEnqueueReleaseGLObjects(queue, 1, &output_image, 0, NULL, NULL);
    error_check(err);

    clFinish(queue);
}

void draw_texture() {
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program);

    GLint brightness_uniform_location = glGetUniformLocation(program, "brightness");
    glUniform1f(brightness_uniform_location, 1.0f + mouse_wheel_y * 0.01f);

    glActiveTexture(GL_TEXTURE0 + MAIN_TEXTURE_UNIT_INDEX);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindSampler(MAIN_TEXTURE_UNIT_INDEX, sampler);

    glBindBuffer(GL_ARRAY_BUFFER, position_buffer_obj);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (const void *) (16 * sizeof(float)));

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindSampler(MAIN_TEXTURE_UNIT_INDEX, sampler);
    glUseProgram(0);
}

template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
    return std::max(lower, std::min(n, upper));
}

void mouse_motion(int x, int y) {
    if (mouse_active_click == GLUT_LEFT_BUTTON) {
        if (!slice_view_mode) {
            transformation_x += x - mouse_left_x;
            transformation_y += y - mouse_left_y;
            slice_number = 0;
        }

        mouse_left_x = x;
        mouse_left_y = y;
    } else if (mouse_active_click == GLUT_RIGHT_BUTTON) {
        slice_number += y - mouse_right_y;
        slice_number = clip(slice_number, 0, depth - 1);

        mouse_right_y = y;
    }
}

void mouse_action(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        mouse_active_click = GLUT_LEFT_BUTTON;
        mouse_left_x = x;
        mouse_left_y = y;
    } else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        mouse_active_click = GLUT_RIGHT_BUTTON;
        mouse_right_x = x;
        mouse_right_y = y;
    } else if (button == 3 && state == GLUT_DOWN) {
        mouse_wheel_y++;
    } else if (button == 4 && state == GLUT_DOWN) {
        mouse_wheel_y--;
    }
}

void keyboard_action(unsigned char key, int x, int y) {
    if (key == ' ') {
        slice_view_mode = !slice_view_mode;
    }
}

namespace po = boost::program_options;

po::variables_map read_variables(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "prints this message")
            ("all-devices,a", "prints all available devices")
            ("platform,p", po::value<int>()->default_value(0), "number of platform to use")
            ("device,d", po::value<int>()->default_value(0), "number of device to use")
            ("input-file", po::value<vector<string>>(), "input files");

    po::positional_options_description pos_desc;
    pos_desc.add("input-file", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos_desc).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc;
        exit(EXIT_SUCCESS);
    }

    return vm;
}

int main(int argc, char* argv[]) {

    const po::variables_map &options = read_variables(argc, argv);

    if (options.count("all-devices")) {
        print_info();
        exit(EXIT_SUCCESS);
    }

    if (options["input-file"].empty()) {
        error("No files to read");
    }

    vector<string> input_files = options["input-file"].as<vector<string>>();

    int original_width = 0;
    int original_height = 0;
    int original_depth = 0;

    for (int i = 0; i < input_files.size(); i++) {

        DicomImage *image = new DicomImage(input_files[i].c_str());

        if (i == 0) {
            original_width = (int) image->getWidth();
            original_height = (int) image->getHeight();
        } else {
            if (original_width != image->getWidth() || original_height != image->getHeight()) {
                error("Image %s expected size is %dx%d but was %dx%d", input_files[i].c_str(), original_width, original_height, image->getWidth(),
                      image->getHeight());
            }
        }

        original_depth += (int) image->getFrameCount();

        delete image;
    }

    cout << "Original size is " << original_width << "x" << original_height << "x" << original_depth << endl;

    int n = max(max(original_width, original_height), original_depth); // cube n x n x n
    n = round_up(n, 4); // we use uchar4

    width = n;
    height = n;
    depth = n;

    unsigned char * buffer = new unsigned char[width * height * depth ];
    unsigned char * buffer_cursor = buffer + ((width - original_width) / 2 + ((height - original_height) / 2) * width +
            ((depth - original_depth) / 2) * width * height);
    unsigned char * slice_buffer = new unsigned char [original_width * original_height];

    for (int i = 0; i < input_files.size(); i++) {
        DicomImage *image = new DicomImage(input_files[i].c_str());

        for (int j = 0; j < image->getFrameCount(); j++) {
            image->getOutputData(slice_buffer, (const unsigned long) (original_width * original_height), 8, (const unsigned long) j);

            for (int k = 0; k < original_height; k++) {
                memcpy(buffer_cursor, slice_buffer + k * original_width, original_width);
                buffer_cursor += width;
            }
            buffer_cursor += width * (height - original_height);
        }
        cout << input_files[i] << endl;
        delete image;
    }

    cout << "Rendered size is " << width << "x" << height << "x" << depth << endl;

    init_glut(argc, argv);

    init_gl();

    cl_platform_id platform = get_platform(options["platform"].as<int>());
    cl_device_id device = get_device(platform, options["device"].as<int>());
    cl_context context = get_context(platform, device);

    queue = clCreateCommandQueue(context, device, NULL, &err);
    error_check(err);

    // INPUT_IMAGE
    cl_mem input_image = create_input_image_object(context, buffer);

    size_t input_image_row_pitch;
    size_t input_image_slice_pitch;
    void * input_mem = clEnqueueMapImage(queue, input_image, CL_TRUE, CL_MAP_READ, new size_t[3] {0, 0, 0}, new
            size_t[3]
                    {width, height, depth}, &input_image_row_pitch, &input_image_slice_pitch, 0, NULL, NULL, &err);
    error_check(err);



    // TRANSFORMATION_BUFFER
    unsigned char *transformation_buffer_array = new unsigned char[width * height * depth];
    cl_mem transformation_buffer = create_transformation_buffer(context, transformation_buffer_array);

    void *transformation_mem = clEnqueueMapBuffer(queue, transformation_buffer, CL_TRUE, CL_MAP_READ,
                                                 0, sizeof(unsigned char) * width * depth * height, 0,
                                                 NULL, NULL, &err);
    error_check(err);

    // OUTPUT IMAGE
    float *output_array = new float[width * height];
    output_image = create_output_image_object(context, output_array);

    cl_program program = get_cl_program(context, RAYTRACING_KERNEL, TRANSFORMATION_KERNEL, SLICING_KERNEL);

    build_cl_program(program, device);

    raytracing_kernel = clCreateKernel(program, "raytracing", &err);
    error_check(err);
    transformation_kernel = clCreateKernel(program, "transformation", &err);
    error_check(err);
    slicing_kernel = clCreateKernel(program, "slicing", &err);
    error_check(err);

    // WORK GROUP SIZES
    size_t prefered_work_size;
    clGetKernelWorkGroupInfo(raytracing_kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t),
                             &prefered_work_size, NULL);
    size_t max_work_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_size, NULL);

    default_local_work_size = new size_t[3]{prefered_work_size, 1, 1};
    default_global_work_size = new size_t[3]{round_up(width / 4, default_local_work_size[0]), height, depth};

    size_t raytracing_worksize_z = max_work_size / prefered_work_size;
    if (raytracing_worksize_z < 2) {
        raytracing_local_work_size = new size_t[3]{max_work_size / 2, 1, 2};
    } else {
        raytracing_local_work_size = new size_t[3]{prefered_work_size, 1, raytracing_worksize_z};
    }

    slicing_global_work_size = new size_t[3]{round_up(width / 4, default_local_work_size[1]), height, 1};

    print_work_size("Default local worksize is ", default_local_work_size);
    print_work_size("Default global worksize is ", default_global_work_size);
    print_work_size("Raytracing local worksize is ", raytracing_local_work_size);
    print_work_size("Slicing global worksize is ", slicing_global_work_size);

    // TRANSFORMATION_KERNEL
    err = clSetKernelArg(transformation_kernel, 0, sizeof(cl_mem), &input_image);
    error_check(err);

    err = clSetKernelArg(transformation_kernel, 1, sizeof(cl_mem), &transformation_buffer);
    error_check(err);

    int input_image_size[4] = {width, height, depth, 1 };
    err = clSetKernelArg(transformation_kernel, 2, sizeof(input_image_size), input_image_size);
    error_check(err);

    // RAYTRACING KERNEL
    err = clSetKernelArg(raytracing_kernel, 0, sizeof(cl_mem), &transformation_buffer);
    error_check(err);

    int transformation_buffer_size[4] = {width, height, depth, 1 };
    err = clSetKernelArg(raytracing_kernel, 1, sizeof(transformation_buffer_size), transformation_buffer_size);
    error_check(err);

    err = clSetKernelArg(raytracing_kernel, 2, sizeof(unsigned char) * raytracing_local_work_size[0] * 4 * raytracing_local_work_size[1] *
            raytracing_local_work_size[2], NULL);
    error_check(err);

    // SLICING KERNEL
    err = clSetKernelArg(slicing_kernel, 0, sizeof(cl_mem), &transformation_buffer);
    error_check(err);

    err = clSetKernelArg(slicing_kernel, 1, sizeof(cl_mem), &output_image);
    error_check(err);

    err = clSetKernelArg(slicing_kernel, 2, sizeof(transformation_buffer_size), transformation_buffer_size);
    error_check(err);

    clFinish(queue);

    glutMainLoop();

    err = clEnqueueUnmapMemObject(queue, input_image, input_mem, 0, NULL, NULL);
    error_check(err);

    err = clEnqueueUnmapMemObject(queue, transformation_buffer, transformation_mem, 0, NULL, NULL);
    error_check(err);

    clFinish(queue);

    return EXIT_SUCCESS;
}

void print_work_size(const char *message, size_t *work_size) {
    cout << message << work_size[0] << "x" << work_size[1] << "x" << work_size[2] << endl;
}

//*************************
//GL STUFF
//*************************

void init_glut(int &argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitContextVersion(2,0);
    glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow("Spookfish");
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    glutDisplayFunc(display);
    glutMouseFunc(mouse_action);
    glutMotionFunc(mouse_motion);
    glutKeyboardFunc(keyboard_action);
}

void init_gl() {
    glClearColor(0.0, 0.0, 0.0, 1.0);

    glewInit();

    GLuint vertex_shader = create_shader(GL_VERTEX_SHADER, VERTEX_SHADER);
    GLuint fragment_shader = create_shader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
    program = create_gl_program(vertex_shader, fragment_shader);

    glUseProgram(program);
    GLint texture_uniform_location = glGetUniformLocation(program, "colorTexture");
    glUniform1i(texture_uniform_location, MAIN_TEXTURE_UNIT_INDEX);
    glUseProgram(0);

    glGenBuffers(1, &position_buffer_obj);

    glBindBuffer(GL_ARRAY_BUFFER, position_buffer_obj);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_positions), vertex_positions, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    texture = create_texture();
    sampler = create_sampler();
}

GLuint create_gl_program(GLuint vertex_shader, GLuint fragment_shader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);

    if (status == GL_FALSE)
    {
        GLint log_length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);

        GLchar *log_buffer = new GLchar[log_length + 1];
        glGetProgramInfoLog(program, log_length, NULL, log_buffer);
        error("Linker failure: %s\n", log_buffer);
    }
    glDetachShader(program, vertex_shader);
    glDetachShader(program, fragment_shader);
    return program;
}

GLuint create_shader(GLenum shader_type, const string shader_file) {
    string source = read_file(shader_file);
    GLuint shader = glCreateShader(shader_type);
    char const *source_c_str = source.c_str();
    unsigned long size = source.size();
    glShaderSource(shader, 1, &source_c_str, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLint log_length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
        GLchar *log_buffer = new GLchar[log_length + 1];
        glGetShaderInfoLog(shader, log_length, NULL, log_buffer);
        error("Shader build failed\n%s\n", log_buffer);
    }
    return shader;
}

GLuint create_texture() {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    float *texture_data = new float[4 * width * height];
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, texture_data);
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

GLuint create_sampler() {
    GLuint sampler;
    glGenSamplers(1, &sampler);

    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    return sampler;
}

//*************************
//CL STUFF
//*************************

cl_platform_id get_platform(unsigned int platform_num) {
    cl_uint num_platforms;
    clGetPlatformIDs(1, NULL, &num_platforms);

    if (platform_num >= num_platforms) {
        error("Number of available platforms is %d, cannot obtain plafrom %d\n", num_platforms, platform_num);
    }

    cl_platform_id * platforms = new cl_platform_id[num_platforms];
    clGetPlatformIDs(num_platforms, platforms, NULL);
    return platforms[platform_num];
}

cl_device_id get_device(const cl_platform_id platform, unsigned int device_num) {
    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

    if (device_num >= num_devices) {
        error("Number of available devices is %d, cannot obtaint device %d\n", num_devices, device_num);
    }

    cl_device_id * devices = new cl_device_id[num_devices];
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    return devices[device_num];
}

cl_context get_context(const cl_platform_id platform, const cl_device_id device) {
#ifdef __linux__
    cl_context_properties properties[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
#else
    cl_context_properties properties[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
#endif

    cl_context context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
    error_check(err, "Unable to create context");
    return context;
}

cl_program get_cl_program(const cl_context context, const string filename, const string filename2, const string filename3) {
    string file_contents = read_file(filename);
    string file_contents2 = read_file(filename2);
    string file_contents3 = read_file(filename3);

    const char * c_str[3] = {file_contents.c_str(), file_contents2.c_str(), file_contents3.c_str()};

    size_t file_sizes[3] = {file_contents.size(), file_contents2.size(), file_contents3.size()};
    cl_program program = clCreateProgramWithSource(context, 3, c_str, file_sizes, &err);
    error_check(err, "Unable to create program");

    return program;
};

void build_cl_program(const cl_program program, const cl_device_id device) {
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,  0, NULL, &log_size);

        char * log = new char[log_size + 1];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';

        error("Program build failed\n%s\n", log);
    }
}

cl_mem create_output_image_object(const cl_context context, const float * output_array) {

    cl_mem output_image_obj = clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D,
                                                  0, texture, &err);
    error_check(err, "Error creating image from texture");
    return output_image_obj;
}

cl_mem create_input_image_object(const cl_context context, const unsigned char *input_image_array) {
    cl_image_format gs_format;
    gs_format.image_channel_data_type = CL_UNSIGNED_INT8;
    gs_format.image_channel_order = CL_R;

    cl_image_desc input_img_desc;
    input_img_desc.image_width = width;
    input_img_desc.image_height = height;
    input_img_desc.image_depth = depth;
    input_img_desc.image_row_pitch = sizeof(unsigned char) * width;
    input_img_desc.image_slice_pitch = sizeof(unsigned char) * width * height;
    input_img_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
    input_img_desc.image_array_size = 1;
    input_img_desc.num_mip_levels = NULL;
    input_img_desc.num_samples = NULL;
    input_img_desc.buffer = NULL;

    cl_mem input_img_obj = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR , &gs_format,
                                         &input_img_desc, (void *) input_image_array, &err);
    error_check(err, "Error creating 3D image object");

    return input_img_obj;
}

cl_mem create_transformation_buffer(const cl_context context, const unsigned char *transformation_buffer_array) {
    cl_mem transformation_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                                sizeof(unsigned char) * width * height * depth, (void *)
                                                        transformation_buffer_array, &err);
    error_check(err, "Error creating transformation buffer object");
    return transformation_buffer;
}

//*************************
//GENERAL STUFF
//*************************
streamsize get_stream_size(ifstream &input_file) {
    input_file.seekg(0, input_file.end);
    streamsize size = input_file.tellg();
    input_file.seekg(0, input_file.beg);
    return size;
}

vector<char> read_file_to_buffer(const string filename) {
    ifstream input_file(filename, ifstream::binary);
    streamsize size = get_stream_size(input_file);

    vector<char> buffer;
    buffer.reserve(size);
    buffer.assign(istreambuf_iterator<char>(input_file), istreambuf_iterator<char>());
    input_file.close();
    return buffer;
}

string read_file(const string filename) {
    ifstream input_file(filename);
    streamsize size = get_stream_size(input_file);

    if (size < 0) {
        error("Cannot read file %s\n", filename.c_str());
    }
    string str;
    str.reserve(size + 1);
    str.assign(istreambuf_iterator<char>(input_file), istreambuf_iterator<char>());
    str.push_back('\0');
    input_file.close();
    return str;
}

vector<float> to_floats(const vector<char> data) {
    vector<float> floats;
    floats.reserve(data.size());
    for_each(data.begin(), data.end(), [&](char c) {
        floats.push_back((float) c / (float) CHAR_MAX);
    });
    return floats;
}

void error(const string error_message, ...) {
    va_list args;
    va_start (args, error_message);
    vfprintf(stderr, error_message.c_str(), args);
    va_end (args);
    exit(EXIT_FAILURE);
}

void error_check(cl_int err, const string error_message, ...) {
    if (err != CL_SUCCESS) {
        va_list args;
        va_start (args, error_message);
        fprintf(stderr, "Error %d\n", err);
        error(error_message, args);
        va_end (args);
    }
}

void error_check(cl_int err) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void print_time_diff(clock_t start) {
    start = clock();
    cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
}


void print_info() {
    cl_int err;

    cl_uint num_platforms;
    err = clGetPlatformIDs(1, NULL, &num_platforms);
    printf("Number of platform is %d\n", num_platforms);

    cl_platform_id *platforms;
    platforms = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, NULL);


    for (unsigned int i = 0; i < num_platforms; i++) {
        cl_platform_id platform = platforms[i];

        cl_platform_info platform_infos[] = {CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS};
        for (unsigned int j = 0; j < sizeof(platform_infos) / sizeof(cl_platform_info); j++) {
            size_t param_size;
            err = clGetPlatformInfo(platform, platform_infos[j], 0, NULL, &param_size);

            char *param_value = (char *) malloc(param_size);
            err = clGetPlatformInfo(platform, platform_infos[j], param_size, param_value, NULL);

            printf("Platform %d param %d: %s\n", i, j, param_value);

        }

        cl_uint num_devices;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices); // 0 or 1 num_entries
        printf("Number of devices is %d\n", num_devices);

        cl_device_id *devices;
        devices = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        for (unsigned int k = 0; k < num_devices; k++) {
            cl_device_id device = devices[k];

            cl_device_info device_infos[] = {CL_DEVICE_NAME, CL_DEVICE_VENDOR, CL_DEVICE_EXTENSIONS, CL_DEVICE_OPENCL_C_VERSION};
            for (unsigned int l = 0; l < sizeof(device_infos) / sizeof(cl_device_info); l++) {
                size_t param_size;
                err = clGetDeviceInfo(device, device_infos[l], 0, NULL, &param_size);

                char *param_value = (char *) malloc(param_size);
                err = clGetDeviceInfo(device, device_infos[l], param_size, param_value, NULL);
                printf("Device %d param %d: %s\n", k, l, param_value);
            }

            cl_ulong global_mem_size;
            err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
            printf("Device %d CL_DEVICE_GLOBAL_MEM_SIZE: %lu\n", k, global_mem_size);

            cl_uint address_bits;
            err = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &address_bits, NULL);
            printf("Device %d CL_DEVICE_ADDRESS_BITS: %u\n", k, address_bits);

            cl_bool available;
            err = clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, NULL);
            printf("Device %d CL_DEVICE_AVAILABLE: %u\n", k, available);

            cl_bool compiler_available;
            err = clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &compiler_available, NULL);
            printf("Device %d CL_DEVICE_COMPILER_AVAILABLE: %u\n", k, compiler_available);

            cl_bool image_support;
            err = clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &image_support, NULL);
            printf("Device %d CL_DEVICE_IMAGE_SUPPORT: %u\n", k, image_support);

            size_t max_work_group_size;
            err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
            printf("Device %d CL_DEVICE_MAX_WORK_GROUP_SIZE: %u\n", k, max_work_group_size);

            cl_uint max_work_item_dimensions;
            err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dimensions, NULL);
            printf("Device %d CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: %u\n", k, max_work_item_dimensions);

            size_t *max_work_item_sizes = (size_t *) malloc(3 * sizeof(size_t));
            err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof(size_t), max_work_item_sizes, NULL);
            printf("Device %d CL_DEVICE_MAX_WORK_ITEM_SIZES: %u, %u, %u\n", k, max_work_item_sizes[0],
                   max_work_item_sizes[1], max_work_item_sizes[2]);

        }
    }

}
