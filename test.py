import weggli

code = """
#include <iostream>
#include <cstring> // for memcpy

int main() {
    // Define a source array
    int sourceArray[] = {10, 20, 30, 40, 50};
    int a = 1;
    size_t numElements = sizeof(sourceArray) / sizeof(sourceArray[0]);
    size_t bufferSize = numElements * sizeof(int);

    // Allocate a buffer with the same size as sourceArray
    int* buffer = new int[numElements];
    buffer[2] = buffer[3];

    // Use memcpy to copy the contents of sourceArray into the buffer
    memcpy(buffer, sourceArray, bufferSize);

    // Print the contents of the buffer to verify the copy
    std::cout << "Buffer contents: ";
    for (size_t i = 0; i < numElements; ++i) {
        std::cout << buffer[i] << " ";
    }
    std::cout << std::endl;

    // Free the allocated memory
    delete[] buffer;

    return 0;
}
"""

qt = weggli.parse_query("{int $a = _+foo+$a;}", False)
print(weggli.identifiers(qt))
print(weggli.matches(qt, "void foo() {int bar=10+foo+bar;}"))