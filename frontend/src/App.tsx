import { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  CloudArrowUpIcon, EyeIcon, MoonIcon, SunIcon, 
  ArrowPathIcon, ClipboardDocumentCheckIcon 
} from '@heroicons/react/24/outline';

const App = () => {
  const [jobFile, setJobFile] = useState<File | null>(null);
  const [cvFiles, setCvFiles] = useState<File[]>([]);
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const handleProcess = async () => {
    if (!jobFile || cvFiles.length === 0) return alert("Pilih file kualifikasi dan CV!");
    setLoading(true);
    
    const formData = new FormData();
    formData.append('job_file', jobFile);
    cvFiles.forEach(f => formData.append('cv_files', f));

    try {
      // Path relatif agar bekerja dengan rewrite vercel.json
      const res = await axios.post('/api/match', formData);
      setResults(res.data.results);
    } catch (e) {
      console.error(e);
      alert("Gagal memproses data. Pastikan backend berjalan.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-screen bg-slate-50 dark:bg-slate-950 transition-colors overflow-x-hidden text-slate-900 dark:text-white">
      <header className="h-20 bg-white dark:bg-slate-900 border-b dark:border-slate-800 flex items-center justify-between px-8 sticky top-0 z-50 shadow-sm">
        <div className="flex items-center gap-4">
          <img src="/logo_ptoba.png" className="h-10 w-auto bg-white p-1 rounded shadow-sm" alt="Logo" />
          <h1 className="text-xl font-bold tracking-tight">Recruitment Matcher</h1>
        </div>
        <button onClick={() => setIsDarkMode(!isDarkMode)} className="p-2.5 bg-slate-100 dark:bg-slate-800 rounded-full">
          {isDarkMode ? <SunIcon className="w-6 h-6 text-yellow-500" /> : <MoonIcon className="w-6 h-6 text-slate-600" />}
        </button>
      </header>

      <main className="max-w-7xl mx-auto p-8 space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white dark:bg-slate-900 p-10 rounded-3xl border-2 border-dashed border-slate-200 dark:border-slate-800 text-center relative group hover:border-blue-500 transition-colors">
            <CloudArrowUpIcon className="w-14 h-14 mx-auto text-blue-500 mb-4" />
            <h3 className="font-bold">Kualifikasi Pekerjaan (PDF)</h3>
            <input type="file" accept=".pdf" className="absolute inset-0 opacity-0 cursor-pointer" onChange={(e) => setJobFile(e.target.files?.[0] || null)} />
            <p className="text-sm text-slate-500 mt-3">{jobFile ? `✔ ${jobFile.name}` : "Klik untuk upload"}</p>
          </div>

          <div className="bg-white dark:bg-slate-900 p-10 rounded-3xl border-2 border-dashed border-slate-200 dark:border-slate-800 text-center relative group hover:border-indigo-500 transition-colors">
            <CloudArrowUpIcon className="w-14 h-14 mx-auto text-indigo-500 mb-4" />
            <h3 className="font-bold">Unggah CV (Banyak PDF)</h3>
            <input type="file" multiple accept=".pdf" className="absolute inset-0 opacity-0 cursor-pointer" onChange={(e) => setCvFiles(Array.from(e.target.files || []))} />
            <p className="text-sm text-slate-500 mt-3">{cvFiles.length > 0 ? `✔ ${cvFiles.length} CV terpilih` : "Klik untuk upload"}</p>
          </div>
        </div>

        <button onClick={handleProcess} disabled={loading} className="w-full py-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-2xl flex justify-center items-center gap-4 disabled:bg-slate-400">
          {loading ? <ArrowPathIcon className="w-7 h-7 animate-spin" /> : <ClipboardDocumentCheckIcon className="w-7 h-7" />}
          <span>{loading ? "Menganalisis..." : "Mulai Ranking"}</span>
        </button>

        {results.length > 0 && (
          <div className="bg-white dark:bg-slate-900 rounded-3xl shadow-xl overflow-hidden border dark:border-slate-800">
            <table className="w-full text-left">
              <thead className="bg-slate-50 dark:bg-slate-800 text-xs uppercase font-bold text-slate-500">
                <tr>
                  <th className="p-5">Rank</th>
                  <th className="p-5">Nama Kandidat</th>
                  <th className="p-5 text-center">Skor</th>
                  <th className="p-5 text-center">Status</th>
                  <th className="p-5 text-center">Aksi</th>
                </tr>
              </thead>
              <tbody className="divide-y dark:divide-slate-800">
                {results.map((res, i) => (
                  <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                    <td className="p-5 font-bold">{i + 1}</td>
                    <td className="p-5">{res.filename}</td>
                    <td className="p-5 text-center text-blue-600 font-bold">{res.percentage}%</td>
                    <td className="p-5 text-center">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${res.status === 'Cocok' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {res.status}
                      </span>
                    </td>
                    <td className="p-5 text-center">
                      <button 
                        onClick={() => {
                          const file = cvFiles.find(f => f.name === res.filename);
                          if(file) window.open(URL.createObjectURL(file));
                        }}
                        className="p-2 bg-slate-100 dark:bg-slate-800 rounded-lg hover:bg-blue-500 hover:text-white transition-colors"
                      >
                        <EyeIcon className="w-5 h-5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
      <footer className="text-center py-10 text-slate-400 text-sm">© 2026 PT Oba Bersama Abadi</footer>
    </div>
  );
};

export default App;