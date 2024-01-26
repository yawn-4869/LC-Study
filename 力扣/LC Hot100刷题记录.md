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



## 17. 电话号码的字母组合

