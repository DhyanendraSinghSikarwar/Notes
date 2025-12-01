# Linux Command Notes

## Basic System Information

-   **uname** -- Print system information
    -   `uname -a` -- Print all system info\
    -   `man uname` -- Show manual\
-   **hostname** -- Print system/host name\
-   **whoami** -- Show current logged-in user

## Sudo & User Switching

-   **sudo** -- Run command as root
    -   `sudo <cmd>`\
    -   `sudo -i` -- Login as root\
    -   `sudo su -` -- Switch to root\
    -   `su <username>` -- Switch user\
-   **tty** -- Show terminal file name

## System Monitoring

-   **top** -- Live process view
    -   `q` exit, `M` sort by memory, `P` sort by CPU\
-   **df -h** -- Disk usage (human readable)\
-   **free -h** -- Memory usage

------------------------------------------------------------------------

# Frequently Asked Linux Commands

1.  ls\
2.  cd\
3.  pwd\
4.  mkdir\
5.  rm, rmdir\
6.  cat, zcat\
7.  touch\
8.  head\
9.  tail, `tail -f`\
10. less, more\
11. cp\
12. mv\
13. wc\
14. vi editor\
15. ln (hard link, soft link)\
16. cut\
17. tee\
18. sort\
19. clear\
20. diff\
21. ssh\
22. df\
23. du\
24. ps\
25. top\
26. fuser\
27. kill\
28. nohup\
29. free\
30. vmstat\
31. nproc -- Check CPU count

------------------------------------------------------------------------

# System-Level Commands

1.  uname\
2.  uptime\
3.  date\
4.  who, whoami\
5.  which\
6.  id\
7.  sudo\
8.  shutdown\
9.  reboot\
10. apt\
11. yum\
12. dnf\
13. pacman\
14. portage

------------------------------------------------------------------------

# User & Group Management Commands

1.  sudo\
2.  useradd\
3.  whoami\
4.  su\
5.  passwd\
6.  userdel\
7.  groupadd\
8.  gpasswd `-a`, `-m`\
9.  groupdel

------------------------------------------------------------------------

# File Permission Commands

1.  umask\
2.  ls -l\
3.  chmod\
4.  chown\
5.  chgrp

------------------------------------------------------------------------

# Compression Commands

1.  zip, gunzip, gzip\
2.  tar, untar

------------------------------------------------------------------------

# File Transfer Commands

1.  scp -- Secure copy\
2.  rsync -- Remote sync

------------------------------------------------------------------------

# Additional Useful Commands

-   `ps -ef` -- Show all running processes\
-   `curl <url>` -- Hit a URL\
-   `sudo find / -name <file>` -- Search file\
-   `wget <url>` -- Download file

------------------------------------------------------------------------

# Shell Script Options

-   `set -x` -- Show commands before execution\
-   `set -e` -- Stop execution on error\
-   `set -o pipefail` -- Show error from piped commands
