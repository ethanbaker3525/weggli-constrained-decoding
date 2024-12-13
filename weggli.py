import subprocess
import tempfile
import os

def test(a):
    print(a)
    return False
    #return "J" in a

def weggli(query, source_code: str) -> str:

    try:
        # Write the source code to a temporary file with a recognized C++ extension
        with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as source_file:
            source_file.write(source_code)
            source_file_path = os.path.abspath(source_file.name)

        # Use STDIN to provide the filename to Weggli
        command = "weggli " + " ".join(query) + " -"
        #print(command)
        
        result = subprocess.run(
            command,
            input=source_file_path + "\n",  # Pipe the file path to Weggli
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Weggli error: {result.stderr}")

        #print(result.stderr)
        return result.stdout
    finally:
        if 'source_file_path' in locals() and os.path.exists(source_file_path):
            os.unlink(source_file_path)

def multi_weggli(queries, source_code: str) -> str:
    return [weggli(query, source_code) for query in queries]
    


def weggli_query_constraint(q):
    def f(src):
        print(src)
        return weggli(q, src) != ""
    return f

def weggli_queries_constraint(qs):
    def f(src):
        print(src)
        for q in qs:
            if weggli(q, src) != "":
                return True
        return False
    return f

if __name__ == "__main__":

    # Example usage
    query = """
    {
        memcpy(_,_,6);
    }
    """
    source_code = """
```%
<｜begin▁of▁sentence｜>
              
Complete the following code using `memcpy`.

```
#include <stdio.h>
#include <string.h>
              
int main() {

    char str1[] = "Hello ";
    char str2[] = "there!";

    puts("str1 before memcpy ");
    puts(str1);

    // Copies contents of str2 to str1

    memcpy(str1, str2, 5);

    puts("\nstr1 after memcpy");
    puts(str1);
}

    """
    output = weggli(query, source_code)
    print(output == "")
