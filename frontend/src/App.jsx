import { useState, useEffect, useCallback } from 'react'
import './App.css'

// Use relative URL when served from FastAPI, localhost for dev
const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : ''

function App() {
  const [constants, setConstants] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedConstant, setSelectedConstant] = useState(null)
  const [format, setFormat] = useState('json')
  const [rawResponse, setRawResponse] = useState('')
  const [loading, setLoading] = useState(false)

  const fetchConstants = useCallback(async (query = '') => {
    setLoading(true)
    try {
      const url = query
        ? `${API_BASE}/constants?q=${encodeURIComponent(query)}`
        : `${API_BASE}/constants`
      const response = await fetch(url, {
        headers: { 'Accept': 'application/json' }
      })
      const data = await response.json()
      setConstants(data)
    } catch (error) {
      console.error('Error fetching constants:', error)
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchConstants()
  }, [fetchConstants])

  const fetchConstantDetail = async (symbol) => {
    setLoading(true)
    const mimeTypes = {
      json: 'application/json',
      html: 'text/html',
      turtle: 'text/turtle',
      n3: 'text/n3',
      jsonld: 'application/ld+json'
    }

    try {
      const response = await fetch(`${API_BASE}/constants/${symbol}`, {
        headers: { 'Accept': mimeTypes[format] }
      })
      const text = await response.text()
      setRawResponse(text)

      if (format === 'json') {
        setSelectedConstant(JSON.parse(text))
      } else {
        setSelectedConstant({ symbol, raw: true })
      }
    } catch (error) {
      console.error('Error fetching constant:', error)
    }
    setLoading(false)
  }

  const handleSearch = (e) => {
    e.preventDefault()
    fetchConstants(searchQuery)
  }

  return (
    <div className="app">
      <header>
        <h1>🔬 Drum API Explorer</h1>
        <p>Browse and search physical fundamental constants</p>
      </header>

      <main>
        <section className="search-section">
          <form onSubmit={handleSearch}>
            <input
              type="text"
              placeholder="Search constants (e.g., planck, speed, electron)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <button type="submit">Search</button>
            <button type="button" onClick={() => { setSearchQuery(''); fetchConstants(); }}>
              Clear
            </button>
          </form>
        </section>

        <div className="content">
          <section className="constants-list">
            <h2>Constants {loading && '(loading...)'}</h2>
            <ul>
              {constants.map((c) => (
                <li
                  key={c.symbol}
                  onClick={() => fetchConstantDetail(c.symbol)}
                  className={selectedConstant?.symbol === c.symbol ? 'selected' : ''}
                >
                  <strong>{c.name}</strong>
                  <code>{c.symbol}</code>
                </li>
              ))}
            </ul>
          </section>

          <section className="detail-section">
            <div className="format-selector">
              <label>Response format:</label>
              <select value={format} onChange={(e) => setFormat(e.target.value)}>
                <option value="json">JSON</option>
                <option value="html">HTML</option>
                <option value="turtle">Turtle (RDF)</option>
                <option value="n3">N3 (RDF)</option>
                <option value="jsonld">JSON-LD</option>
              </select>
              {selectedConstant && (
                <button onClick={() => fetchConstantDetail(selectedConstant.symbol)}>
                  Refresh
                </button>
              )}
            </div>

            {selectedConstant && (
              <div className="detail-content">
                <h2>
                  {selectedConstant.raw
                    ? `${selectedConstant.symbol} (${format})`
                    : `${selectedConstant.name} (${selectedConstant.symbol})`
                  }
                </h2>

                {!selectedConstant.raw && format === 'json' && (
                  <div className="detail-card">
                    <p><strong>Value:</strong> {selectedConstant.value}</p>
                    {selectedConstant.valueDecimal && <p><strong>Value (Decimal):</strong> {selectedConstant.valueDecimal}</p>}
                    {selectedConstant.valueFloat && <p><strong>Value (Float):</strong> {selectedConstant.valueFloat}</p>}
                    <p><strong>Uncertainty:</strong> {selectedConstant.uncertainty || '—'}</p>
                    {selectedConstant.uncertaintyDecimal && <p><strong>Uncertainty (Decimal):</strong> {selectedConstant.uncertaintyDecimal}</p>}
                    {selectedConstant.uncertaintyFloat && <p><strong>Uncertainty (Float):</strong> {selectedConstant.uncertaintyFloat}</p>}
                    <p><strong>Unit:</strong> {selectedConstant.unit || '—'}</p>
                  </div>
                )}

                <div className="raw-response">
                  <h3>Raw API Response:</h3>
                  <pre>{rawResponse}</pre>
                </div>

                <div className="api-example">
                  <h3>Try it yourself:</h3>
                  <code>
                    curl -H "Accept: {format === 'json' ? 'application/json' :
                      format === 'html' ? 'text/html' :
                        format === 'turtle' ? 'text/turtle' :
                          format === 'n3' ? 'text/n3' : 'application/ld+json'}" \<br />
                    &nbsp;&nbsp;{API_BASE}/constants/{selectedConstant.symbol}
                  </code>
                </div>
              </div>
            )}

            {!selectedConstant && (
              <div className="placeholder">
                <p>👈 Select a constant to view details</p>
                <p>Try different response formats to see content negotiation in action!</p>
              </div>
            )}
          </section>
        </div>
      </main>

      <footer>
        <p>
          API Documentation: <a href={`${API_BASE}/docs`} target="_blank">/docs</a> |
          <a href={`${API_BASE}/redoc`} target="_blank">/redoc</a>
        </p>
      </footer>
    </div>
  )
}

export default App
