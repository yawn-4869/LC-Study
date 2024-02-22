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



## 153. 寻找旋转排序数组中的最小值 20240129

```c++
class Solution {
public:
    int findMin(vector<int>& nums) {
        int n = nums.size();
        int left = 0, right = n-1;
        while(left <= right) {
            int mid = left + (right - left) / 2;
            if(nums[mid] > nums[n-1]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return nums[left];
    }
};
```



## 20. 有效的括号 20240129

```c++
class Solution {
public:
    bool isValid(string s) {
        stack<char> stk;
        for(char& c : s) {
            if(c == '(' || c == '{' || c == '[') {
                stk.push(c);
            } else {
                if (c == ')') {
                    if(stk.empty()) {
                        return false;
                    } else {
                        if(stk.top() != '(') {
                            return false;
                        }
                        stk.pop();
                    }
                }

                if (c == '}') {
                    if(stk.empty()) {
                        return false;
                    } else {
                        if(stk.top() != '{') {
                            return false;
                        }
                        stk.pop();
                    }
                }

                if (c == ']') {
                    if(stk.empty()) {
                        return false;
                    } else {
                        if(stk.top() != '[') {
                            return false;
                        }
                        stk.pop();
                    }
                }
            }
        }
        return stk.empty();
    }
};
```



## 155. 最小栈 20240129

* 使用额外空间的解法

```c++
class MinStack {
    stack<int> min_stk;
    stack<int> x_stk;
public:
    MinStack() {
        min_stk.push(INT_MAX);
    }
    
    void push(int val) {
        x_stk.push(val);
        min_stk.push(min(min_stk.top(), val));
    }
    
    void pop() {
        x_stk.pop();
        min_stk.pop();
    }
    
    int top() {
        return x_stk.top();
    }
    
    int getMin() {
        return min_stk.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(val);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
```



* 不使用额外空间

```c++
class MinStack {
private:
    stack<long long> stk; // stk用于保存当前栈最小值与当前元素未压入栈时的差值
    long long min_val; // 存储最小值
public:
    MinStack() {

    }
    
    void push(int val) {
        if(stk.empty()) {
            stk.push(0LL);
            min_val = (long long)val;
            return;
        }

        stk.push((long long)val - min_val);
        min_val = min(min_val, (long long)val);
    }
    
    void pop() {
        if(stk.top() <= 0) {
            min_val -= stk.top();
        }
        stk.pop();
    }
    
    int top() {
        return stk.top() < 0LL ? (int)min_val : (int)min_val + stk.top();
    }
    
    int getMin() {
        return (int)min_val;
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(val);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
```



## 394. 字符串解码 20240222

* 思路：使用辅助栈来解决字符串中的括号嵌套问题

  对于字符串 `s` 中的字符 `c` ，有以下四种情况

  * `c 为字符` 直接加入到 `res` 中
  * `c 为数字` 记录到 `multi` 
  * ` c == [ ` 将`res` 和 `multi` 入栈，并将 `res` 置空，`multi` 置 `0` 
  * `c == ]` 将先前存入的 `last_res` 和 `last_multi` 出栈，`res = last_res + res * last_multi`

```c++
class Solution {
public:
    string decodeString(string s) {
        int n = s.size();
        string res;
        int multi = 0;
        stack<pair<int, string>> stk;
        for (int i = 0; i < n; ++i) {
            if (s[i] >= 'a' && s[i] <= 'z') {
                // 直接添加
                res.push_back(s[i]);
            }
            else if (s[i] >= '0' && s[i] <= '9') {
                // 记录倍数
                multi = multi * 10 + (s[i] - '0');
            }
            else if (s[i] == '[') {
                // 入栈
                stk.push(make_pair(multi, res));
                res = "";
                multi = 0;
            }
            else {
                // 出栈
                auto p = stk.top();
                stk.pop();
                string last_res = p.second;
                for (int j = 0; j < p.first; ++j) {
                    last_res.append(res);
                }
                res = last_res;
            }
        }
        return res;
    }
};
```



## 739. 每日温度 20240222

* 思路：使用单调栈，从右至左遍历

  对于数组中的第`i` 个元素，栈中已经存入了数组 `[i+1, n-1]` 区间的元素

  对于第 `i-1` 个元素，若 `[i+1, n-1]` 中存在比 `temperatures[i]` 小且比 `temperatures[i-1]` 大的元素，那 `temperatures[i]` 比大于 `temperatures[i-1]` ，`ans[i-1] = 1`

  因此对于每个元素 `temperatures[i]` ，我们每次都 `pop` 出比它小的元素

```c++
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        stack<pair<int, int>> stk;
        vector<int> ans(n);
        // 从右往左遍历
        for(int i = n-1; i >= 0; --i) {
            while(!stk.empty() && stk.top().second <= temperatures[i]) {
                stk.pop();
            }
            ans[i] = stk.empty() ? 0 : stk.top().first - i;
            stk.push({i, temperatures[i]});
        }
        return ans;
    }
};
```



## 215. 数组中第k个最大元素 20240222

```c++
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        int l = 0, r = nums.size() - 1;
        while(l < r) {
            int i = l, j = r;
            int pivot = nums[i];
            while(i < j) {
                while(i < j && nums[j] < pivot) j--;
                if(i < j) {
                    swap(nums[i], nums[j]);
                }
                while(i < j && nums[i] >= pivot) i++;
                if(i < j) {
                    swap(nums[i], nums[j]);
                }
            }
            if(i == k-1) {
                break;
            } else if(i < k-1) {
                l = i+1;
            } else {
                r = i-1;
            }
        }
        return nums[k-1];
    }
};
```



## 347. 前k个高频元素 20240222

* 思路：使用哈希表进行计数，再通过优先队列来进行统计

```c++
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> nums_cnt;
        for(auto &num : nums) {
            nums_cnt[num]++;
        }
        // 优先队列, 根据p.first降序排序
        priority_queue<pair<int, int>, vector<pair<int, int>>> max_heap;
        for(auto ite = nums_cnt.begin(); ite != nums_cnt.end(); ite++) {
            max_heap.emplace(make_pair(ite->second, ite->first));
        }
        vector<int> ans;
        while(k--) {
            ans.push_back(max_heap.top().second);
            max_heap.pop();
        }
        return ans;
    }
};
```



## 121. 买卖股票的最佳时机 20240222

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int min_price = INT_MAX, max_profit = 0;
        for(auto &price : prices) {
            min_price = min(min_price, price);
            max_profit = max(max_profit, price - min_price);
        }
        return max_profit;
    }
};
```

