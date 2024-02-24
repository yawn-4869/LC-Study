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



## 55. 跳跃游戏 20240223

* 思路：将其转化为最大可到达范围

```c++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int cover = 0;
        if(nums.size() == 1) return true;
        for(int i = 0; i <= cover; ++i) {
            cover = max(i+nums[i], cover);
            if(cover >= nums.size() - 1) return true;
        }
        return false;
    }
};
```



## 45. 跳跃游戏II 20240223

* 思路：对于每次跳跃，找到其跳跃范围[start, end]，进行跳跃模拟

```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        // if(nums.size() == 1) return 0;
        int end = 0; // 本次跳跃终点
        int start = 0; // 本次跳跃起点
        int max_pos = 0; // 跳跃边界
        int ans = 0;
        while(end < nums.size() - 1) {
            for(int i = start; i <= end; ++i) {
                // 在本次跳跃范围内循环找到下一次跳跃终点
                max_pos = max(max_pos, i+nums[i]);
            }
            // 迭代
            start = end;
            end = max_pos;
            ans++;
        }
        return ans;
    }
};
```



## 763. 划分字母区间 20240223

* 思路：对于字符串中的每个字母 `c` ，找到其在字符串中最后出现的位置 `last[c]` 

  由于同一字母最多出现在一个片段中，因此分割的子字符串的长度为 `max(end, last[c])` 

  循环遍历，得到每个子字符串的长度

```c++
class Solution {
public:
    vector<int> partitionLabels(string s) {
        vector<int> last(26);
        vector<int> ans;
        int n = s.size();
        for(int i = 0; i < n; ++i) {
            last[s[i] - 'a'] = i;
        }

        int start = 0, end = 0;
        for(int i = 0; i < n; ++i) {
            end = max(end, last[s[i] - 'a']);
            if(i == end) {
                ans.push_back(end - start + 1);
                start = end + 1;
            }
        }
        return ans;
    }
};
```



## 70. 爬楼梯 20240223

```c++
class Solution {
public:
    int climbStairs(int n) {
        // f[i] = f[i-1] + f[i-2]
        // f[0] = 1 f[1] = 1
        int first = 1, second = 1;
        for(int i = 2; i <= n; ++i) {
            int tmp = second;
            second = first + second;
            first = tmp;
        }
        return second;
    }
};
```



## 118. 杨辉三角 20240223

```c++
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> ans(numRows);
        for(int i = 0; i < numRows; ++i) {
            ans[i].resize(i+1, 1);
            // ans[i][j] = ans[i-1][j-1] + ans[i-1][j]
            if(i == 0) continue;
            for(int j = 1; j < i; ++j) {
                ans[i][j] = ans[i-1][j-1] + ans[i-1][j];
            }
        }
        return ans;
    }
};
```



## 198. 打家劫舍 20240223

```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        // f[i] = max(f[i-1], f[i-2] + nums[i])
        if(nums.size() == 1) return nums[0];
        int first = nums[0], second = max(nums[0], nums[1]);
        for(int i = 2; i < nums.size(); ++i) {
            int tmp = second;
            second = max(second, first + nums[i]);
            first = tmp;
        }
        return second;
    }
};
```



## 139. 单词拆分 20240223

```c++
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> word_dict_set;
        for(auto &word : wordDict) {
            word_dict_set.emplace(word);
        }

        int n = s.size();
        vector<bool> dp(n+1);
        dp[0] = true;
        for(int i = 1; i <= n; ++i) {
            for(int j = 0; j < i; ++j) {
                if(dp[j] && word_dict_set.find(s.substr(j, i - j)) != word_dict_set.end()) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }
};
```



## 300. 最长递增子序列 20240224

* 思路（O(n^2)）：记数组 `dp[i]` 为以 `nums[i]` 结尾的数组最长升序子序列的长度

对于 `dp[i]` 包含的区间 `[0, i-1]` 中的数字 `j` , 若 `nums[i] > nums[j]` , 则有：

`dp[i] = max(dp[j]) + 1` 否则，`dp[i] = max(dp[j])` 

对于整个数组，找到 `dp` 的最大值即可

```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n);
        for(int i = 0; i < n; ++i) {
            dp[i] = 1; // 对于每个子数组dp[i], nums[i]必为其元素
            for(int j = 0; j < i; ++j) {
                if(nums[i] > nums[j]) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};
```



## 152. 乘积最大子数组 20240224

* 思路：可以得到简单的递推关系：`f[i] = max(f[i-1]*nums[i], nums[i]` 

  但考虑当前位置 `nums[i]` , 当 `nums[i]` 为负数时，我们需要前面的 `f[i-1]` 应当是尽可能地小（尽可能地负的更多）

  因此，需要维护两个数组 `max_f` 和 `min_f` 

  得到递推关系：

  `max_f[i] = max(max_f[i-1] * nums[i], max(min_f[i-1] * nums[i], nums[i])) `

  `min_f[i] = min(min_f[i-1] * nums[i], min(max_f[i-1] * nums[i], nums[i]))` 

```c++
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        vector<int> max_f(nums), min_f(nums);
        for(int i = 1; i < nums.size(); ++i) {
            max_f[i] = max(max_f[i-1] * nums[i], max(nums[i], min_f[i-1] * nums[i]));
            min_f[i] = min(min_f[i-1] * nums[i], min(nums[i], max_f[i-1] * nums[i]));
        }
        return *max_element(max_f.begin(), max_f.end());
    }
};
```



## 62. 不同路径 20240224

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n));
        // dp[i][j] = dp[i-1][j] + dp[i][j-1]
        dp[0][0] = 1;
        for(int i = 0; i < m; ++i) {
            for(int j = 0; j < n; ++j) {
                if(i - 1 >= 0) {
                    dp[i][j] += dp[i-1][j];
                }
                if(j - 1 >= 0) {
                    dp[i][j] += dp[i][j-1];
                }
            }
        }
        return dp[m-1][n-1];
    }    
};
```



## 64. 最小路径和 20240224

```c++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        for(int i = 0; i < m; ++i) {
            for(int j = 0; j < n; ++j) {
                if(i - 1 < 0 && j - 1 < 0) {
                    dp[i][j] = grid[i][j];
                } else if(i - 1 < 0) {
                    dp[i][j] = dp[i][j-1] + grid[i][j];
                } else if(j - 1 < 0) {
                    dp[i][j] = dp[i-1][j] + grid[i][j];
                } else {
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
                }
            }
        }
        return dp[m-1][n-1];
    }
};
```



## 5. 最长回文子串

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size();
        int l = 0, r = 0;
        for(int i = 0; i < n; ++i) {
            auto [s1, e1] = getPosOfPalindrome(s, i, i);
            auto [s2, e2] = getPosOfPalindrome(s, i, i+1);
            if(e1 - s1 > r - l) {
                l = s1; 
                r = e1;
            }

            if(e2 - s2 > r - l) {
                l = s2;
                r = e2;
            }
        }
        return s.substr(l, r - l + 1);
    }

    pair<int, int> getPosOfPalindrome(string s, int l, int r) {
        int n = s.size();
        while(l >= 0 && r < n) {
            if(s[l] != s[r]) {
                break;
            }
            l--;
            r++;
        }
        return {l+1, r-1};
    }
};
```

