#include <list>
#include <unordered_map>

template <typename K, typename V> class LRUCache
{
private:
    using Key = K;
    using Value = V*;
    using List = std::list<std::pair<Key, Value>>;
    using Map = std::unordered_map<Key, typename List::iterator>;

    List list;
    Map map;
    size_t capacity;

public:
    LRUCache(size_t cap = 50) : capacity(cap) {}

    bool find(const Key& key)
    {
        auto it = map.find(key);
        if (it == map.end())
            return false;
        return true;
    }

    Value get(const Key& key)
    {
        auto it = map.find(key);
        if (it == map.end())
            return nullptr;
        list.splice(list.begin(), list, it->second);
        return it->second->second;
    }

    void put(const Key& key, const Value value)
    {
        auto it = map.find(key);
        if (it != map.end())
        {
            it->second->second = value;
            list.splice(list.begin(), list, it->second);
            return;
        }
        list.emplace_front(key, value);
        map[key] = list.begin();
        if (list.size() > capacity)
        {
            auto last = list.end();
            last--;
            delete last->second;
            map.erase(last->first);
            list.pop_back();
        }
    }
};