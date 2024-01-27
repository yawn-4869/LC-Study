## 994. 腐烂的橘子 20240116

* 广度优先搜索
* 使用二维数组`dis[x][y]`来保存访问到每个结点时候的距离
* 使用`cnt`来保存新鲜橘子的数量，便于最后判定是否全部腐烂

```c++
class Solution {
    int direction[4][2] = {{0,1},{0,-1},{1,0},{-1,0}};
    int dis[10][10];
public:
    int orangesRotting(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        queue<pair<int, int>> Q;
        memset(dis, -1, sizeof(dis));
        int cnt = 0; // 记录新鲜橘子的数量
        int ans = 0; // 记录时间
        for(int i = 0; i < m; ++i) {
            for(int j = 0; j < n; ++j) {
                if(grid[i][j] == 2) {
                    Q.push({i, j});
                    dis[i][j] = 0;
                } else if(grid[i][j] == 1){
                    cnt++;
                }
            }
        }
        while(!Q.empty()) {
            auto p = Q.front();
            Q.pop();
            int x = p.first, y = p.second;
            // if(x+1 < m && grid[x+1][y] == 1) {
            //     grid[x+1][y] = 2;
            //     Q.push({x+1, y});
            //     cnt--;
            // }
            // if(x-1 >= 0 && grid[x-1][y] == 1) {
            //     grid[x-1][y] = 2;
            //     Q.push({x-1, y});
            //     cnt--;
            // }
            // if(y+1 < n && grid[x][y+1] == 1){
            //     grid[x][y+1] = 2;
            //     Q.push({x, y+1});
            //     cnt--;
            // }
            // if(y-1 >= 0 && grid[x][y-1] == 1) {
            //     grid[x][y-1] = 2;
            //     Q.push({x, y-1});
            //     cnt--;
            // }
            for(int i = 0; i < 4; ++i) {
                int nx = x + direction[i][0];
                int ny = y + direction[i][1];
                if(nx < 0 || nx >= m || ny < 0 || ny >= n || !grid[nx][ny] || ~dis[nx][ny])
                    continue;
                dis[nx][ny] = dis[x][y] + 1;
                Q.push({nx, ny});
                if(grid[nx][ny] == 1) {
                    cnt--;
                    ans = dis[nx][ny];
                    if(!cnt) break;
                }
            }
        }
        return cnt ? -1 : ans;
    }
};
```



## 207. 课程表 20240116

* 拓扑排序

```c++
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        int n = prerequisites.size();
        int cnt = 0;
        vector<int> indegree(numCourses);
        vector<vector<int>> g(numCourses);
        for(int i = 0; i < n; ++i) {
            indegree[prerequisites[i][0]]++;
        }
        for(int i = 0; i < n; ++i) {
            g[prerequisites[i][1]].push_back(prerequisites[i][0]);
        }
        queue<int> Q;
        for(int i = 0; i < indegree.size(); ++i) {
            if(indegree[i] == 0) {
                Q.push(i);
            }
        }
        if(Q.empty()) return false;
        while(!Q.empty()) {
            int x = Q.front();
            Q.pop();
            cnt++;
            for(int& e : g[x]) {
                indegree[e]--;
                if(indegree[e] == 0) {
                    Q.push(e);
                }
            }
        }
        return cnt == numCourses;
    }
};
```



## 46. 全排列 20240116

* 回溯

```c++
class Solution {
    vector<vector<int>> ans;
    vector<bool> visited;
public:
    vector<vector<int>> permute(vector<int>& nums) {
        visited.resize(nums.size());
        vector<int> temp;
        backtrack(nums, temp);
        return ans;
    }
    void backtrack(vector<int>& nums, vector<int>& temp) {
        if(temp.size() == nums.size()) {
            ans.push_back(temp);
            return;
        }

        for(int i = 0; i < nums.size(); ++i) {
            if(visited[i]) continue;
            temp.push_back(nums[i]);
            visited[i] = true;
            backtrack(nums, temp);
            temp.pop_back();
            visited[i] = false;
        }

    }
};
```

## 78. 子集 20240117

* 迭代法
* 对于数组中的每个数，只有两种状态：被选中和不被选中
* 使用`mask`来进行状态的存储，如对于数组`[1,2,3]`, `mask = 1 =>[1], mask = 3 => [1,2]    `

```c++
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        int n = nums.size();
        vector<int> tmp;
        vector<vector<int>> ans;
        for(int mask = 0; mask < (1 << n); ++mask) {
            tmp.clear();
            for(int i = 0; i < n; ++i) {
                if(mask & (1 << i)) {
                    tmp.push_back(nums[i]);
                }
            }
            ans.push_back(tmp);
        }
        return ans;
    }
};
```



* 回溯法

```c++
class Solution {
public:
    vector<int> tmp;
    vector<vector<int>> ans;

    void dfs(int cur, vector<int>& nums) {
        if(cur == nums.size()) {
            ans.push_back(tmp);
            return;
        }
        tmp.push_back(nums[cur]);
        dfs(cur+1, nums);
        tmp.pop_back();
        dfs(cur+1, nums);
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        dfs(0, nums);
        return ans;
    }
};
```



## 17. 电话号码的字母组合 20240126

```c++
class Solution {
    vector<string> ans;
public:
    vector<string> letterCombinations(string digits) {
        if(digits.empty()) {
            return ans;
        }
        // 初始化map
        unordered_map<char, string> index;
        index['2'] = "abc";
        index['3'] = "def";
        index['4'] = "ghi";
        index['5'] = "jkl";
        index['6'] = "mno";
        index['7'] = "pqrs";
        index['8'] = "tuv";
        index['9'] = "wxyz";

        string path = "";
        backtrack(index, digits, path, 0);
        return ans;
    }

    void backtrack(unordered_map<char, string>& index, string digits, string& path, int i) {
        if(i == digits.size()) {
            ans.push_back(path);
            return;
        }
        string str = index[digits[i]];
        for(char& c : str) {
            path.push_back(c);
            backtrack(index, digits, path, i+1);
            path.pop_back();
        }
    }
};
```



## 39. 组合总和 20240126

```c++
class Solution {
    vector<vector<int>> ans;
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<int> path;
        backtrack(candidates, path, target, 0);
        return ans;
    }
    void backtrack(vector<int>& candidates, vector<int>& path, int target, int index) {
        if(index == candidates.size() || target <= 0) {
            if(target == 0) {
                ans.push_back(path);
            }
            return;
        }

        path.push_back(candidates[index]);
        backtrack(candidates, path, target - candidates[index], index);
        path.pop_back();
        backtrack(candidates, path, target, index + 1);

    }
};
```



## 22. 括号生成 20240126

```c++
class Solution {
    vector<string> ans;
public:
    vector<string> generateParenthesis(int n) {
        // path要初始化，以防backtrack的时候访问越界
        string path(2*n, '(');
        backtrack(n, 0, path);
        return ans;
    }

    void backtrack(int n, int index, string& path) {
        if(index == 2 * n) {
            if(isValid(path)) {
                ans.push_back(path);
            }
            return;
        }
        path[index] = '(';
        backtrack(n, index + 1, path);
        path[index] = ')';
        backtrack(n, index + 1, path);
    }

    bool isValid(string parenthesis) {
        stack<char> stk;
        for(char &c : parenthesis) {
            if(c == '(') {
                stk.push(c);
            } else {
                if(stk.empty()) {
                    return false;
                }
                stk.pop();
            }
        }
        return stk.empty();
    }
};
```



## 79. 单词搜索 20240126

```c++
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        stack<pair<int, int>> stk;
        int m = board.size(), n = board[0].size();
        for(int i = 0; i < m; ++i) {
            for(int j = 0; j < n; ++j) {
                if(backtrack(board, i, j, 0, word)) return true;
            }
        }
        return false;
    }

    bool backtrack(vector<vector<char>>& board, int x, int y, int index, string word) {
        if(index == word.size()) {
            return true;
        }
        int m = board.size(), n = board[0].size();
        if(x < 0 || x >= m || y < 0 || y >= n || board[x][y] != word[index]) {
            return false;
        }
        board[x][y] = '\0';
        bool res = backtrack(board, x+1, y, index+1, word) || backtrack(board, x-1, y, index+1, word) || 
                    backtrack(board, x, y+1, index+1, word) || backtrack(board, x, y-1, index+1, word);
        board[x][y] = word[index];
        return res;
    }
};
```



## 131. 分割回文串 20240127

```c++
class Solution {
    vector<vector<string>> ans;
public:
    vector<vector<string>> partition(string s) {
        int n = s.size();
        // 预处理，减少重复比较
        vector<vector<bool>> f(n, vector<bool>(n, true));
        for(int i = n-1; i >= 0; --i) {
            for(int j = i+1; j < n; ++j) {
                f[i][j] = (s[i] == s[j]) && f[i+1][j-1];
            }
        }

        vector<string> path;
        backtrack(s, 0, path, f);
        return ans;
    }

    void backtrack(string s, int idx, vector<string>& path, vector<vector<bool>> f) {
        if(idx == s.size()) {
            ans.push_back(path);
            return;
        }

        for(int j = idx; j < s.size(); ++j) {
            if(f[idx][j]) {
                path.push_back(s.substr(idx, j-idx+1));
                backtrack(s, j+1, path, f);
                path.pop_back();
            }
        }
    }
};
```



## 35. 搜索插入位置 20240127

```c++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while(left <= right) {
            int mid = left + (right - left) / 2;
            if(nums[mid] == target) {
                return mid;
            } else if(nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }
};
```



## 74. 搜索二维矩阵 20240127

```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int i = 0, j = n - 1;
        while(i < m && j >= 0) {
            if(matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] < target) {
                i++;
            } else {
                j--;
            }
        }
        return false;
    }
};
```



## 34. 在排序数组中查找元素的第一个和最后一个位置 20240127

```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        int pos = -1;
        while(left <= right) {
            int mid = left + (right - left) / 2;
            if(nums[mid] == target) {
                pos = mid;
                break;
            } else if(nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if(pos == -1) {
            return {-1, -1};
        }
        left = right = pos;
        while(left >= 0 && nums[left] == target) left--;
        while(right < nums.size() && nums[right] == target) right++;
        return {left+1, right-1};
    }
};
```



## 33. 搜索旋转排序数组 20240127

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int n = nums.size();
        // 根据二分查找k的位置
        int left = 0, right = n - 1;
        while(left <= right) {
            int mid = left + (right - left) / 2;
            if(nums[mid] > nums[n-1]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        int k = left;
        
        // 两次二分查找
        left = 0, right = k - 1;
        while(left <= right) {
            int mid = left + (right - left) / 2;
            if(nums[mid] == target) {
                return mid;
            } else if(nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        left = k, right = n - 1;
        while(left <= right) {
            int mid = left + (right - left) / 2;
            if(nums[mid] == target) {
                return mid;
            } else if(nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return -1;
    }
};
```



