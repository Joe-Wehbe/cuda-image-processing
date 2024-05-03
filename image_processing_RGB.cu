#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <sys/time.h>

typedef struct
{
    int height;
    int width;
    int pixel_size;
    png_infop info_ptr;
    png_byte *buf;
} PNG_RAW;

long long timeInMilliseconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (((long long)tv.tv_sec) * 1000) + (tv.tv_usec / 1000);
}

PNG_RAW *read_png(char *file_name)
{
    PNG_RAW *png_raw = (PNG_RAW *)malloc(sizeof(PNG_RAW));
    FILE *fp = fopen(file_name, "rb");
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);
    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    int pixel_size = 3;
    png_raw->width = width;
    png_raw->height = height;
    png_raw->pixel_size = pixel_size;
    png_raw->buf = (png_byte *)malloc(width * height * pixel_size * sizeof(png_byte));
    png_raw->info_ptr = info_ptr;
    int k = 0;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width * pixel_size; j++)
        {
            png_raw->buf[k++] = row_pointers[i][j];
        }
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    fclose(fp);
    return png_raw;
}

void write_png(char *file_name, PNG_RAW *png_raw)
{
    FILE *fp = fopen(file_name, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_init_io(png_ptr, fp);
    png_infop info_ptr = png_raw->info_ptr;
    int width = png_raw->width;
    int height = png_raw->height;
    int pixel_size = png_raw->pixel_size;
    png_bytepp row_pointers;
    row_pointers = (png_bytepp)malloc(height * sizeof(png_bytep));
    for (int i = 0; i < height; i++)
        row_pointers[i] = (png_bytep)malloc(width * pixel_size);
    int k = 0;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width * pixel_size; j++)
        {
            row_pointers[i][j] = png_raw->buf[k++];
        }

    png_set_rows(png_ptr, info_ptr, row_pointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    for (int i = 0; i < height; i++)
        free(row_pointers[i]);
    free(row_pointers);
    fclose(fp);
}

void process_on_host(PNG_RAW *png_raw)
{
    long long start = timeInMilliseconds();
    for (int i = 0; i < png_raw->width * png_raw->height; i++)
    {
        int index = i * png_raw->pixel_size;
        png_raw->buf[index] = (png_byte)255; // Red channel
        png_raw->buf[index + 1] = (png_byte)0; // Green channel
        png_raw->buf[index + 2] = (png_byte)0; // Blue channel
    }
    long long end = timeInMilliseconds();
    printf("Timing on host is %lld millis\n", end - start);
}

__global__ void BlurKernel(png_byte *d_P, int height, int width, int pixel_size)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = (Row * width + Col) * pixel_size;

    if (Row < height && Col < width)
    {
        int sumR = 0, sumG = 0, sumB = 0;
        int count = 0;

        for (int i = -2; i <= 2; i++)
        {
            for (int j = -2; j <= 2; j++)
            {
                int curRow = Row + i;
                int curCol = Col + j;

                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                {
                    int curIndex = (curRow * width + curCol) * pixel_size;
                    sumR += d_P[curIndex];
                    sumG += d_P[curIndex + 1];
                    sumB += d_P[curIndex + 2];
                    count++;
                }
            }
        }
        png_byte avgR = sumR / count;
        png_byte avgG = sumG / count;
        png_byte avgB = sumB / count;

        d_P[index] = avgR;
        d_P[index + 1] = avgG;
        d_P[index + 2] = avgB;
    }
}

void process_blurring_on_device(PNG_RAW *png_raw)
{
    int m = png_raw->height;
    int n = png_raw->width;
    int pixel_size = png_raw->pixel_size;

    png_byte *d_P;
    cudaError_t err;

    long long start = timeInMilliseconds();

    err = cudaMalloc((void **)&d_P, m * n * pixel_size * sizeof(png_byte));
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_P, png_raw->buf, m * n * pixel_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    BlurKernel<<<gridDim, blockDim>>>(d_P, m, n, pixel_size);

    cudaMemcpy(png_raw->buf, d_P, m * n * pixel_size * sizeof(png_byte), cudaMemcpyDeviceToHost);

    long long end = timeInMilliseconds();

    printf("\n");
    printf("Blurring your image... \n");
    printf("Timing on Device is %lld millis\n", end - start);

    cudaFree(d_P);
}

__global__ void EdgeDetectionKernel(png_byte *d_P, int height, int width, int pixel_size)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = (Row * width + Col) * pixel_size;

    if (Row < height && Col < width)
    {
        int GxR = 0, GxG = 0, GxB = 0;
        int GyR = 0, GyG = 0, GyB = 0;

        int sobelMaskX[3][3] = { { -1, 0, 1 },
                                 { -2, 0, 2 },
                                 { -1, 0, 1 } };

        int sobelMaskY[3][3] = { { -1, -2, -1 },
                                 { 0, 0, 0 },
                                 { 1, 2, 1 } };

        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                int curRow = Row + i;
                int curCol = Col + j;

                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                {
                    int curIndex = (curRow * width + curCol) * pixel_size;
                    int maskValueX = sobelMaskX[i + 1][j + 1];
                    int maskValueY = sobelMaskY[i + 1][j + 1];

                    GxR += d_P[curIndex] * maskValueX;
                    GxG += d_P[curIndex + 1] * maskValueX;
                    GxB += d_P[curIndex + 2] * maskValueX;

                    GyR += d_P[curIndex] * maskValueY;
                    GyG += d_P[curIndex + 1] * maskValueY;
                    GyB += d_P[curIndex + 2] * maskValueY;
                }
            }
        }

        int gradientMagnitudeR = sqrtf(GxR * GxR + GyR * GyR);
        int gradientMagnitudeG = sqrtf(GxG * GxG + GyG * GyG);
        int gradientMagnitudeB = sqrtf(GxB * GxB + GyB * GyB);

        png_byte normalizedMagnitudeR = (png_byte)(gradientMagnitudeR / 255.0f * 255.0f);
        png_byte normalizedMagnitudeG = (png_byte)(gradientMagnitudeG / 255.0f * 255.0f);
        png_byte normalizedMagnitudeB = (png_byte)(gradientMagnitudeB / 255.0f * 255.0f);

        d_P[index] = normalizedMagnitudeR;
        d_P[index + 1] = normalizedMagnitudeG;
        d_P[index + 2] = normalizedMagnitudeB;
    }
}

void process_edge_detection_on_device(PNG_RAW *png_raw)
{
    int m = png_raw->height;
    int n = png_raw->width;
    int pixel_size = png_raw->pixel_size;

    png_byte *d_P;
    cudaError_t err;

    long long start = timeInMilliseconds();

    err = cudaMalloc((void **)&d_P, m * n * pixel_size * sizeof(png_byte));
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_P, png_raw->buf, m * n * pixel_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    EdgeDetectionKernel<<<gridDim, blockDim>>>(d_P, m, n, pixel_size);

    cudaMemcpy(png_raw->buf, d_P, m * n * pixel_size * sizeof(png_byte), cudaMemcpyDeviceToHost);

    long long end = timeInMilliseconds();

    printf("\n");
    printf("Detecting edges... \n");
    printf("Timing on Device is %lld millis\n", end - start);

    cudaFree(d_P);
}

__global__ void SharpeningKernel(png_byte *d_P, int height, int width, int pixel_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int index = (row * width + col) * pixel_size;

    if (row < height && col < width)
    {
        float center = d_P[index];
        float left = (col > 0) ? d_P[index - pixel_size] : 0.0f;
        float right = (col < width - 1) ? d_P[index + pixel_size] : 0.0f;
        float top = (row > 0) ? d_P[index - width * pixel_size] : 0.0f;
        float bottom = (row < height - 1) ? d_P[index + width * pixel_size] : 0.0f;

        float sharpened = 5.0f * center - (left + right + top + bottom);

        sharpened = fminf(fmaxf(sharpened, 0.0f), 255.0f);

        d_P[index] = (png_byte)sharpened;
    }
}

void process_sharpening_on_device(PNG_RAW *png_raw)
{
    int m = png_raw->height;
    int n = png_raw->width;
    int pixel_size = png_raw->pixel_size;

    png_byte *d_P;
    cudaError_t err;

    long long start = timeInMilliseconds();

    err = cudaMalloc((void **)&d_P, m * n * pixel_size * sizeof(png_byte));
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_P, png_raw->buf, m * n * pixel_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    SharpeningKernel<<<gridDim, blockDim>>>(d_P, m, n, pixel_size);

    cudaMemcpy(png_raw->buf, d_P, m * n * pixel_size * sizeof(png_byte), cudaMemcpyDeviceToHost);

    long long end = timeInMilliseconds();

    printf("\n");
    printf("Sharpening your image... \n");
    printf("Timing on Device is %lld millis\n", end - start);

    cudaFree(d_P);
}

int main(int argc, char **argv)
{
    int on_host = 0;
    int option;

    if (argv[3] != NULL && strcmp(argv[3], "-d") == 0)
        on_host = 0;

    PNG_RAW *png_raw = read_png(argv[1]);
    if (png_raw->pixel_size != 3)
    {
        printf("Error, png file must be on 3 Bytes per pixel\n");
        exit(0);
    }
    else
        printf("RGB Processing for Image of %d x %d pixels\n", png_raw->width, png_raw->height);

    if (on_host){
      process_on_host(png_raw);
    }
    else{

      printf("\n");
      printf("Choose what to do with the image: \n");
      printf("1. Blurring \n");
      printf("2. Edge Detection \n");
      printf("3. Sharpening \n");
      printf("------------------\n");
      printf("Enter your choice: ");

      scanf("%d", &option);

      if (option == 1){
        process_blurring_on_device(png_raw);
      }

      if(option == 2){
        process_edge_detection_on_device(png_raw);
      }

      if(option == 3){
        process_sharpening_on_device(png_raw);
      }

    }   

    write_png(argv[2], png_raw);

    printf("Processing finished\n");

    free(png_raw->buf);
    free(png_raw);

    return 0;
}

