#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <cstring>

int main() {
  int pipefd[2];
  if (pipe(pipefd) == -1) {
    std::cerr << "pipe() failed" << std::endl;
    return 1;
  }

  pid_t pid = fork();
  if (pid == -1) {
    std::cerr << "fork() failed" << std::endl;
    return 1;
  } else if (pid == 0) {
    // Child process writes to pipe
    close(pipefd[0]);
    for (int i = 0; i < 5; i++) {
      const char* message = "hello from child process\n";
      if (write(pipefd[1], message, strlen(message)) == -1) {
        perror("write");
        exit(EXIT_FAILURE);
    }
      std::cout << "child process wrote message: " << message;
      std::cout.flush();
    //   sleep(1);
    }
    // close(pipefd[1]);
    std::cout << "child process finished" << std::endl;
    return 0;
  } else {
    // Parent process also writes to pipe
    close(pipefd[0]);
    for (int i = 0; i < 5; i++) {
      const char* message = "hello from parent process\n";
      write(pipefd[1], message, strlen(message));
      fsync(pipefd[1]);
      std::cout << "parent process wrote message: " << message;
      std::cout.flush();
    //   sleep(1);
    }
    // close(pipefd[1]);
    std::cout << "parent process finished" << std::endl;

    // Wait for child process to finish
    int status;
    waitpid(pid, &status, 0);
    return 0;
  }
}
