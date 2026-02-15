import React from 'react'
import { KnowledgeBaseItem } from './kb/types'
import { useKnowledgeBase } from './kb/useKnowledgeBase'
import { useDocuments } from './kb/useDocuments'
import { useUpload } from './kb/useUpload'
import { useUrlIngest } from './kb/useUrlIngest'
import { KnowledgeBaseHeader } from './kb/KnowledgeBaseHeader'
import { UploadZone } from './kb/UploadZone'
import { UrlIngest } from './kb/UrlIngest'
import { DocumentList } from './kb/DocumentList'

interface KnowledgeBasePanelProps {
  kbId: number
  knowledgeBases: KnowledgeBaseItem[]
  onKbChange: (kbId: number) => void
  onKnowledgeBasesUpdated: () => Promise<void>
}

/**
 * 知识库管理界面：上传文件、导入 URL、查看索引列表。
 */
export const KnowledgeBasePanel: React.FC<KnowledgeBasePanelProps> = ({
  kbId,
  knowledgeBases,
  onKbChange,
  onKnowledgeBasesUpdated,
}) => {
  const {
    creatingKb,
    newKbName,
    setNewKbName,
    newKbDescription,
    setNewKbDescription,
    kbBusy,
    error: kbError,
    setError: setKbError,
    handleCreateKnowledgeBase,
    handleDeleteKnowledgeBase,
    handleReindexKnowledgeBase,
  } = useKnowledgeBase(kbId, onKbChange, onKnowledgeBasesUpdated)

  const {
    docs,
    loading: docsLoading,
    error: docsError,
    setError: setDocsError,
    selectedDocIds,
    setSelectedDocIds,
    loadDocs,
    handleDelete,
    handleReindexDocument,
    handleBatchDelete,
    handleBatchReindex,
  } = useDocuments(kbId)

  const {
    batchUploading,
    uploadTasks,
    duplicatePolicy,
    setDuplicatePolicy,
    error: uploadError,
    setError: setUploadError,
    handleBatchUpload,
    clearFinishedTasks,
  } = useUpload(kbId, docs, loadDocs)

  const {
    urlToIngest,
    setUrlToIngest,
    urlIngesting,
    urlError,
    setUrlError,
    handleIngestUrl,
  } = useUrlIngest(kbId, loadDocs)

  const busy = batchUploading || urlIngesting
  const error = kbError || docsError || uploadError || urlError

  const setError = (msg: string) => {
    if (kbError) setKbError(msg)
    else if (docsError) setDocsError(msg)
    else if (uploadError) setUploadError(msg)
    else if (urlError) setUrlError(msg)
    else setDocsError(msg) // Default
  }

  /**
   * 统计 URL 与文件来源数量。
   */
  const urlCount = docs.filter((doc) => doc.source_type === 'url').length
  const fileCount = docs.length - urlCount

  return (
    <div className="panel kb-panel">
      <KnowledgeBaseHeader
        kbId={kbId}
        knowledgeBases={knowledgeBases}
        onKbChange={onKbChange}
        busy={busy}
        kbBusy={kbBusy}
        handleReindexKnowledgeBase={() => handleReindexKnowledgeBase(loadDocs)}
        handleDeleteKnowledgeBase={handleDeleteKnowledgeBase}
        newKbName={newKbName}
        setNewKbName={setNewKbName}
        newKbDescription={newKbDescription}
        setNewKbDescription={setNewKbDescription}
        creatingKb={creatingKb}
        handleCreateKnowledgeBase={handleCreateKnowledgeBase}
        docsCount={docs.length}
        fileCount={fileCount}
        urlCount={urlCount}
      />

      <div className="kb-actions">
        <UploadZone
          busy={busy}
          batchUploading={batchUploading}
          duplicatePolicy={duplicatePolicy}
          setDuplicatePolicy={setDuplicatePolicy}
          handleBatchUpload={handleBatchUpload}
          uploadTasks={uploadTasks}
          clearFinishedTasks={clearFinishedTasks}
        />

        <UrlIngest
          busy={busy}
          urlToIngest={urlToIngest}
          setUrlToIngest={setUrlToIngest}
          urlIngesting={urlIngesting}
          handleIngestUrl={handleIngestUrl}
        />
      </div>

      <DocumentList
        docs={docs}
        loading={docsLoading}
        error={error}
        loadDocs={loadDocs}
        selectedDocIds={selectedDocIds}
        setSelectedDocIds={setSelectedDocIds}
        handleReindexDocument={handleReindexDocument}
        handleDelete={handleDelete}
        handleBatchReindex={handleBatchReindex}
        handleBatchDelete={handleBatchDelete}
        busy={busy}
        kbBusy={kbBusy}
      />
    </div>
  )
}
