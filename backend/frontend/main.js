document.addEventListener('DOMContentLoaded', () => {
  // --- CONFIGURATION ---
  const API_BASE_URL = 'http://localhost:8000/api/'
  const USER_ID = 'user_123456'

  // --- DOM ELEMENT REFERENCES ---
  const navLinks = document.querySelectorAll('.nav-link')
  const views = document.querySelectorAll('.view')
  const docsTableBody = document.getElementById('docs-table-body')
  const docsTotalCount = document.getElementById('docs-total-count')
  const refreshDocsBtn = document.getElementById('refresh-docs-btn')
  const syncRagBtn = document.getElementById('sync-rag-btn')
  const dropZone = document.getElementById('drop-zone')
  const fileInput = document.getElementById('file-input')
  const filePreviewArea = document.getElementById('file-preview-area')
  const uploadBtn = document.getElementById('upload-btn')
  const cancelUploadBtn = document.getElementById('cancel-upload-btn')
  const queryForm = document.getElementById('query-form')
  const queryInput = document.getElementById('query-input')
  const queryResultsContainer = document.getElementById(
    'query-results-container'
  )

  // --- STATE MANAGEMENT ---
  let filesToUpload = []

  // --- INITIALIZATION ---
  setupEventListeners()
  switchToView('documents-view')
  renderDocumentsTable()

  // --- EVENT LISTENERS ---
  function setupEventListeners() {
    navLinks.forEach((link) => {
      link.addEventListener('click', (e) => {
        e.preventDefault()
        switchToView(link.dataset.view)
      })
    })

    refreshDocsBtn.addEventListener('click', renderDocumentsTable)
    syncRagBtn.addEventListener('click', () => {
      alert('Sync Feature: This requires a backend update to implement.')
    })

    dropZone.addEventListener('click', () => fileInput.click())
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault()
      dropZone.classList.add('dragover')
    })
    dropZone.addEventListener('dragleave', () =>
      dropZone.classList.remove('dragover')
    )
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault()
      dropZone.classList.remove('dragover')
      handleFileSelection(e.dataTransfer.files)
    })

    fileInput.addEventListener('change', (e) =>
      handleFileSelection(e.target.files)
    )
    uploadBtn.addEventListener('click', handleUpload)
    cancelUploadBtn.addEventListener('click', () => {
      filesToUpload = []
      renderFilePreviews()
      switchToView('documents-view')
    })

    queryForm.addEventListener('submit', handleQuerySubmit)

    document.addEventListener('click', (e) => {
      const openMenu = document.querySelector('.action-menu.visible')
      if (openMenu && !openMenu.parentElement.contains(e.target)) {
        openMenu.classList.remove('visible')
      }
    })
  }

  // --- CORE FUNCTIONS ---
  function switchToView(viewId) {
    views.forEach((view) => view.classList.remove('active'))
    navLinks.forEach((link) => link.classList.remove('active'))
    document.getElementById(viewId).classList.add('active')
    document
      .querySelector(`.nav-link[data-view="${viewId}"]`)
      .classList.add('active')
  }

  async function renderDocumentsTable() {
    docsTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 2rem;">Loading documents...</td></tr>`
    try {
      const response = await fetch(`${API_BASE_URL}documents/${USER_ID}`)
      if (!response.ok) throw new Error('Network response was not ok.')

      const data = await response.json()
      docsTableBody.innerHTML = ''

      if (data.documents && data.documents.length > 0) {
        data.documents.forEach((doc) => {
          const row = document.createElement('tr')
          const formattedDate = new Date(doc.date).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          })

          row.innerHTML = `
                        <td><input type="checkbox"></td>
                        <td>${doc.documentname}</td>
                        <td><span class="type-badge">${doc.type}</span></td>
                        <td>${doc.size}</td>
                        <td>${formattedDate}</td>
                        <td class="actions-cell">
                            <button class="action-menu-btn" title="More options">⋮</button>
                            <div class="action-menu">
                                <a href="#" class="action-item" data-action="query"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>Query</a>
                                <a href="#" class="action-item" data-action="preview"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line></svg>Preview</a>
                                <a href="#" class="action-item" data-action="download"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>Download</a>
                                <a href="#" class="action-item action-delete" data-action="delete"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>Delete</a>
                            </div>
                        </td>
                    `
          docsTableBody.appendChild(row)

          const menuButton = row.querySelector('.action-menu-btn')
          const menu = row.querySelector('.action-menu')

          menuButton.addEventListener('click', (e) => {
            e.stopPropagation()
            const isCurrentlyVisible = menu.classList.contains('visible')
            document
              .querySelectorAll('.action-menu.visible')
              .forEach((m) => m.classList.remove('visible'))

            if (!isCurrentlyVisible) {
              const buttonRect = menuButton.getBoundingClientRect()
              if (window.innerHeight - buttonRect.bottom < 160) {
                menu.classList.add('drop-up')
              } else {
                menu.classList.remove('drop-up')
              }
              menu.classList.add('visible')
            }
          })

          menu.querySelectorAll('.action-item').forEach((item) => {
            item.addEventListener('click', (e) => {
              e.preventDefault()
              menu.classList.remove('visible') // Close menu on action
              const action = item.dataset.action

              if (action === 'query') {
                switchToView('query-view')
                queryInput.value = `Tell me about the document "${doc.documentname}".`
              } else if (action === 'delete') {
                if (
                  confirm(
                    `Are you sure you want to delete "${doc.documentname}"?`
                  )
                ) {
                  fetch(
                    `${API_BASE_URL}documents/${USER_ID}/${doc.documentname}`,
                    { method: 'DELETE' }
                  )
                    .then((response) =>
                      response.ok
                        ? response.json()
                        : Promise.reject('Failed to delete.')
                    )
                    .then((data) => {
                      alert(data.detail)
                      renderDocumentsTable()
                    })
                    .catch((error) => {
                      console.error('Delete error:', error)
                      alert('Could not delete the document.')
                    })
                }
              } else if (action === 'preview') {
                window.open(
                  `${API_BASE_URL}documents/preview/${USER_ID}/${doc.documentname}`,
                  '_blank'
                )
              } else if (action === 'download') {
                window.location.href = `${API_BASE_URL}documents/download/${USER_ID}/${doc.documentname}`
              }
            })
          })
        })
      } else {
        docsTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 2rem;">No documents found.</td></tr>`
      }
      docsTotalCount.textContent = `${
        data.total_documents || 0
      } documents total`
    } catch (error) {
      console.error('Failed to fetch documents:', error)
      docsTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 2rem; color: red;">Error loading documents.</td></tr>`
    }
  }

  function handleFileSelection(files) {
    filesToUpload = [...filesToUpload, ...Array.from(files)]
    renderFilePreviews()
  }

  function renderFilePreviews() {
    filePreviewArea.innerHTML = ''
    if (filesToUpload.length > 0) {
      filesToUpload.forEach((file, index) => {
        const item = document.createElement('div')
        item.className = 'file-preview-item'
        item.innerHTML = `<span class="file-name">${file.name}</span><button class="remove-file-btn" title="Remove file">&times;</button>`
        filePreviewArea.appendChild(item)
        item.querySelector('.remove-file-btn').addEventListener('click', () => {
          filesToUpload.splice(index, 1)
          renderFilePreviews()
        })
      })
    }
    uploadBtn.disabled = filesToUpload.length === 0
  }

  async function handleUpload() {
    if (filesToUpload.length === 0) return
    uploadBtn.disabled = true
    uploadBtn.textContent = 'Uploading...'
    filePreviewArea.innerHTML =
      '<p style="text-align: center; color: var(--secondary-text-color);">Upload in progress...</p>'

    const formData = new FormData()
    formData.append('user_id', USER_ID)
    filesToUpload.forEach((file) => formData.append('documents', file))

    try {
      const response = await fetch(`${API_BASE_URL}upload`, {
        method: 'POST',
        body: formData,
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail || 'Upload failed.')

      filePreviewArea.innerHTML = `<p style="text-align: center; color: green;">${
        result.detail || 'Upload complete!'
      }</p>`
      setTimeout(() => {
        filesToUpload = []
        renderFilePreviews()
        switchToView('documents-view')
        renderDocumentsTable()
      }, 2000)
    } catch (error) {
      console.error('Upload error:', error)
      filePreviewArea.innerHTML = `<p style="text-align: center; color: red;">Error: ${error.message}</p>`
    } finally {
      uploadBtn.disabled = false
      uploadBtn.textContent = 'Upload Document'
    }
  }

  async function handleQuerySubmit(event) {
    event.preventDefault()
    const question = queryInput.value.trim()
    if (!question) return
    queryResultsContainer.innerHTML = `<div class="query-answer"><p>Asking AI, please wait...</p></div>`
    try {
      const response = await fetch(`${API_BASE_URL}query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: question,
          user_id: USER_ID,
          stream: false,
        }),
      })
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(
          errorData.detail || 'Failed to get a response from the AI.'
        )
      }
      const data = await response.json()
      queryResultsContainer.innerHTML = `<div class="query-answer"><h3>Answer:</h3><p>${data.content}</p></div>`
    } catch (error) {
      console.error('Query error:', error)
      queryResultsContainer.innerHTML = `<div class="query-answer" style="color: red;"><h3>Error:</h3><p>${error.message}</p></div>`
    }
  }
})
