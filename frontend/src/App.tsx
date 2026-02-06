// frontend/src/App.tsx



import React, { useState, useEffect } from 'react';

import axios from 'axios';

import {

  CloudArrowUpIcon, DocumentTextIcon, EyeIcon, MoonIcon, SunIcon,

  ArrowPathIcon, ClipboardDocumentCheckIcon

} from '@heroicons/react/24/outline';



const App = () => {

  const [jobFile, setJobFile] = useState(null);

  const [cvFiles, setCvFiles] = useState([]);

  const [results, setResults] = useState([]);

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
      // Ganti dengan URL Space kamu
      const res = await axios.post('https://lilcoderi-cv-matcher-app.hf.space/match', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        }
      });
      
      setResults(res.data.results);
    } catch (e) {
      console.error(e);
      alert("Gagal terhubung ke backend Hugging Face!");
    } finally {

      setLoading(false);

    }

  };



  return (

    <div className="min-h-screen w-screen bg-slate-50 dark:bg-slate-950 transition-colors overflow-x-hidden">

     

      {/* HEADER DENGAN LOGO DI KIRI */}

      <header className="h-20 bg-white dark:bg-slate-900 border-b dark:border-slate-800 flex items-center justify-between px-8 sticky top-0 z-50 shadow-sm">

        <div className="flex items-center gap-4">

          <img src="/logo_ptoba.png" className="h-10 w-auto" alt="Logo PT OBA" />

          <h1 className="text-xl font-bold tracking-tight dark:text-white">

            Curriculum Vitae Matcher

          </h1>

        </div>

       

        <button

          onClick={() => setIsDarkMode(!isDarkMode)}

          className="p-2.5 bg-slate-100 dark:bg-slate-800 rounded-full hover:ring-2 ring-blue-500 transition-all"

        >

          {isDarkMode ? (

            <SunIcon className="w-6 h-6 text-yellow-500" />

          ) : (

            <MoonIcon className="w-6 h-6 text-slate-600" />

          )}

        </button>

      </header>



      {/* MAIN CONTENT AREA */}

      <main className="max-w-7xl mx-auto p-8 space-y-8">

       

        {/* UPLOAD CARDS */}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

          {/* Card Job Description */}

          <div className="bg-white dark:bg-slate-900 p-10 rounded-3xl border-2 border-dashed border-slate-200 dark:border-slate-800 text-center relative group hover:border-blue-500 transition-colors">

            <CloudArrowUpIcon className="w-14 h-14 mx-auto text-blue-500 mb-4 group-hover:scale-110 transition-transform" />

            <h3 className="text-lg font-bold dark:text-white transition-colors">

              Kualifikasi Pekerjaan (PDF)

            </h3>

            <input

              type="file"

              accept=".pdf"

              className="absolute inset-0 opacity-0 cursor-pointer"

              onChange={(e) => setJobFile(e.target.files[0])}

            />

            <p className="text-sm text-slate-500 dark:text-slate-400 mt-3 font-medium">

              {jobFile ? `✔ ${jobFile.name}` : "Klik atau seret file PDF kualifikasi"}

            </p>

          </div>



          {/* Card CVs */}

          <div className="bg-white dark:bg-slate-900 p-10 rounded-3xl border-2 border-dashed border-slate-200 dark:border-slate-800 text-center relative group hover:border-indigo-500 transition-colors">

            <CloudArrowUpIcon className="w-14 h-14 mx-auto text-indigo-500 mb-4 group-hover:scale-110 transition-transform" />

            <h3 className="text-lg font-bold dark:text-white transition-colors">

              Unggah CV (Banyak PDF)

            </h3>

            <input

              type="file"

              multiple

              accept=".pdf"

              className="absolute inset-0 opacity-0 cursor-pointer"

              onChange={(e) => setCvFiles(Array.from(e.target.files))}

            />

            <p className="text-sm text-slate-500 dark:text-slate-400 mt-3 font-medium">

              {cvFiles.length > 0 ? `✔ ${cvFiles.length} CV telah dipilih` : "Klik atau seret banyak file PDF CV"}

            </p>

          </div>

        </div>



        {/* PROCESS BUTTON */}

        <button

          onClick={handleProcess}

          disabled={loading}

          className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-2xl shadow-xl shadow-blue-500/20 transition-all flex justify-center items-center gap-4 disabled:bg-slate-400 disabled:shadow-none"

        >

          {loading ? (

            <ArrowPathIcon className="w-7 h-7 animate-spin" />

          ) : (

            <ClipboardDocumentCheckIcon className="w-7 h-7" />

          )}

          <span className="text-lg">

            {loading ? "Sistem Sedang Menganalisis..." : "Mulai Pencocokan & Ranking"}

          </span>

        </button>



        {/* RESULTS TABLE */}

        {results.length > 0 && (

          <div className="bg-white dark:bg-slate-900 rounded-3xl shadow-2xl border border-slate-100 dark:border-slate-800 overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-500">

            <div className="p-6 bg-slate-50 dark:bg-slate-800/50 border-b dark:border-slate-700">

              <h2 className="text-xl font-bold dark:text-white">Hasil Ranking Kandidat</h2>

            </div>

            <div className="overflow-x-auto">

              <table className="w-full text-left border-collapse">

                <thead className="bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-white font-bold uppercase text-xs tracking-wider">

                  <tr>

                    <th className="p-5 border-b dark:border-slate-700">Rank</th>

                    <th className="p-5 border-b dark:border-slate-700">CV Kandidat</th>

                    <th className="p-5 border-b dark:border-slate-700 text-center">Skor Similarity</th>

                    <th className="p-5 border-b dark:border-slate-700 text-center">Status</th>

                    <th className="p-5 border-b dark:border-slate-700 text-center">Aksi</th>

                  </tr>

                </thead>

                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">

                  {results.map((res, i) => (

                    <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors">

                      <td className="p-5">

                        <span className={`inline-flex items-center justify-center w-10 h-10 rounded-xl font-black ${i === 0 ? 'bg-yellow-400 text-white shadow-lg' : 'bg-slate-100 dark:bg-slate-800 dark:text-slate-400 text-slate-500'}`}>

                          {i + 1}

                        </span>

                      </td>

                      <td className="p-5">

                        <div className="flex items-center gap-3 dark:text-slate-100">

                          <DocumentTextIcon className="w-6 h-6 text-slate-400" />

                          {res.filename}

                        </div>

                      </td>

                      <td className="p-5 text-center">

                        <div className="flex flex-col items-center gap-1">

                          <span className="text-lg font-bold text-indigo-600 dark:text-indigo-400">

                            {res.percentage}%

                          </span>

                          <div className="w-20 h-1.5 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">

                            <div className="h-full bg-indigo-500" style={{ width: `${res.percentage}%` }}></div>

                          </div>

                        </div>

                      </td>

                      <td className="p-5 text-center">

                        <span className={`px-4 py-1.5 rounded-full text-xs font-bold tracking-widest uppercase ${res.status === 'Cocok' ? 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-400' : 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-400'}`}>

                          {res.status}

                        </span>

                      </td>

                      <td className="p-5 text-center">

                      <button

                        onClick={() => window.open(URL.createObjectURL(cvFiles.find(f => f.name === res.filename)))}

                        className="p-3 bg-slate-50 dark:bg-slate-800 hover:bg-blue-600 dark:hover:bg-blue-600 hover:text-white dark:text-white rounded-xl transition-all shadow-sm group"

                        title="Preview CV"

                      >

                        {/* Menambahkan text-slate-500 untuk light mode dan dark:text-white untuk dark mode */}

                        <EyeIcon className="w-6 h-6 text-slate-500 dark:text-white group-hover:text-white transition-colors" />

                      </button>

                    </td>

                    </tr>

                  ))}

                </tbody>

              </table>

            </div>

          </div>

        )}

      </main>



      {/* FOOTER */}

      <footer className="text-center py-10 text-slate-400 text-sm">

        © 2026 PT Oba Bersama Abadi - Curriculum Vitae Matcher

      </footer>

    </div>

  );

};



export default App;

