namespace Drawing2Quote.Service.Rules;

/// <summary>
/// A set of non-negative ints that iterates in CPython's order (3.11 set: open addressing, linear probes, then
/// perturbation; a table resized as it fills). The Python reader lists a product's Premium alternatives by iterating
/// such a set, so their order here is the same.
/// </summary>
internal sealed class PyIntSet
{
    private const int LinearProbes = 9, PerturbShift = 5, MinSize = 8;
    private long?[] _table = new long?[MinSize];
    private int _used, _fill;

    public int Count => _used;

    public void Add(long key)
    {
        if (key < 0) throw new ArgumentOutOfRangeException(nameof(key));
        ulong hash = (ulong)key;                     // hash(int) is the int itself (for these)
        int mask = _table.Length - 1;
        ulong perturb = hash;
        int i = (int)(hash & (ulong)mask);
        while (true)
        {
            int probes = i + LinearProbes <= mask ? LinearProbes : 0;
            int j = i;
            do
            {
                var e = _table[j];
                if (e == null)
                {
                    _table[j] = key;
                    _fill++;
                    _used++;
                    if ((long)_fill * 5 >= (long)mask * 3) Resize(_used > 50000 ? _used * 2 : _used * 4);
                    return;
                }
                if (e == key) return;
                j++;
            } while (probes-- > 0);
            perturb >>= PerturbShift;
            i = (int)(((ulong)i * 5 + 1 + perturb) & (ulong)mask);
        }
    }

    private void Resize(int minUsed)
    {
        int size = MinSize;
        while (size <= minUsed) size <<= 1;
        var old = _table;
        _table = new long?[size];
        int mask = size - 1;
        foreach (var e in old)
        {
            if (e == null) continue;
            ulong hash = (ulong)e.Value, perturb = hash;
            int i = (int)(hash & (ulong)mask);
            while (true)
            {
                if (_table[i] == null) { _table[i] = e; break; }
                bool placed = false;
                if (i + LinearProbes <= mask)
                {
                    for (int j = 1; j <= LinearProbes; j++)
                        if (_table[i + j] == null) { _table[i + j] = e; placed = true; break; }
                }
                if (placed) break;
                perturb >>= PerturbShift;
                i = (int)(((ulong)i * 5 + 1 + perturb) & (ulong)mask);
            }
        }
        _fill = _used;
    }

    public IEnumerable<long> Items() => _table.Where(e => e != null).Select(e => e!.Value);
}
