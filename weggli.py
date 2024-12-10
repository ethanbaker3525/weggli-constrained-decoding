import subprocess
import tempfile
import os

def run_weggli(query: str, source_code: str) -> str:

    try:
        # Write the source code to a temporary file with a recognized C++ extension
        with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as source_file:
            source_file.write(source_code)
            source_file_path = os.path.abspath(source_file.name)

        # Use STDIN to provide the filename to Weggli
        command = ["weggli", "--cpp", query, "-"]
        
        result = subprocess.run(
            command,
            input=source_file_path + "\n",  # Pipe the file path to Weggli
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Weggli error: {result.stderr}")

        #print(result.stderr)
        return result.stdout
    finally:
        if 'source_file_path' in locals() and os.path.exists(source_file_path):
            os.unlink(source_file_path)


# Example usage
query = """
{
    _ $buf[_];
    memcpy($buf,_,_);
}
"""
source_code = """
this is a test

#include <stdio.h>
#include <string.h>

void foo(const char *input) {
    char buffer[16]; // Small fixed-size buffer
    printf("Processing input: %s\n", input);

    // Vulnerable: No bounds checking on input
    memcpy(buffer, input, 1);

    printf("Buffer content: %s\n", buffer);
}

"""
output = run_weggli(query, source_code)
print(output)
